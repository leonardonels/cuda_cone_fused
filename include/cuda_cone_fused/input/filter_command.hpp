#pragma once

#include <cuda_cone_fused/ekf_odom.hpp>
#include <cuda_cone_fused/ekf_types.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

/**
 * Command: a reified filter operation. Every sensor — main or
 * optional, motion or correction — turns its adapter output into one of these
 * and drops it on the queue. The filter operation is decoupled from *when* it
 * runs, which is what lets the node apply operations in TIMESTAMP order rather
 * than arrival order.
 */
struct FilterCommand {
  enum class Kind { Predict, Update };

  Kind kind = Kind::Predict;
  uint64_t stamp_ns = 0;

  MotionIncrement motion;            /* valid when kind == Predict */
  std::vector<Observation> obs;      /* valid when kind == Update  */

  static FilterCommand makePredict(const MotionIncrement& m) {
    FilterCommand c;
    c.kind = Kind::Predict;
    c.stamp_ns = m.stamp_ns;
    c.motion = m;
    return c;
  }
  static FilterCommand makeUpdate(std::vector<Observation> o, uint64_t stamp_ns) {
    FilterCommand c;
    c.kind = Kind::Update;
    c.stamp_ns = stamp_ns;
    c.obs = std::move(o);
    return c;
  }
};

/**
 * Timestamp-ordered command queue.
 *
 * Out-of-sequence measurement problem: a cone scan stamped t can arrive after a
 * LIMO msg stamped t+δ. Applying in arrival order would corrupt the filter, so
 * commands are buffered sorted by stamp and applied in stamp order.
 *
 * `drainUpTo(horizon)` applies (and removes) every buffered command with
 * stamp <= horizon, oldest first. The caller sets the horizon to
 * `newestStamp() - late_window` so a SHORT window of the most-recent commands
 * stays buffered, giving a slightly-late older-stamped command the chance to
 * slot in ahead of them before they are applied. A window of 0 applies
 * everything immediately (arrival latency unchanged), still enforcing stamp
 * order among whatever is buffered together.
 */
class CommandQueue {
public:
  struct DrainResult {
    uint32_t predicts = 0;
    uint32_t updates = 0;
    size_t associated = 0;   /* total observations that passed data association */
  };

  void push(FilterCommand cmd) {
    newest_ = std::max(newest_, cmd.stamp_ns);
    buffer_.emplace(cmd.stamp_ns, std::move(cmd));
  }

  DrainResult drainUpTo(uint64_t horizon_ns, EKFOdom& filter) {
    DrainResult r;
    auto it = buffer_.begin();
    while (it != buffer_.end() && it->first <= horizon_ns) {
      const FilterCommand& c = it->second;
      if (c.kind == FilterCommand::Kind::Predict) {
        filter.predict(c.motion);
        ++r.predicts;
      } else {
        r.associated += filter.update(c.obs);
        ++r.updates;
      }
      it = buffer_.erase(it);
    }
    return r;
  }

  uint64_t newestStamp() const { return newest_; }
  size_t size() const { return buffer_.size(); }

private:
  std::multimap<uint64_t, FilterCommand> buffer_;
  uint64_t newest_ = 0;
};
