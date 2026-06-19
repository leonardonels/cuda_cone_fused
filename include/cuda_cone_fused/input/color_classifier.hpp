#pragma once

#include <cuda_cone_fused/color_logic.hpp>

#include <std_msgs/msg/color_rgba.hpp>

#include <cstdint>
#include <memory>

/**
 * Colour classification Strategy + Factory.
 *
 * "One job — map a perception colour to a cone signature — several
 * interchangeable ways, pick ONE at startup." This replaces the
 * `is_colorblind ? yellowCone : getConeColor(...)` ternary with polymorphism
 * (replace conditional with polymorphism), chosen once by the factory from the
 * node parameters.
 *
 * Classification faces PERCEPTION (a single noisy observation). It is orthogonal
 * to ColorLogic's temporal voting, which faces the MAP (per-landmark majority
 * over many observations) and stays where it is.
 */
class IColorClassifier {
public:
  virtual ~IColorClassifier() = default;
  /** Map a single perception colour to a cone signature id (see color_logic.hpp). */
  virtual uint8_t classify(const std_msgs::msg::ColorRGBA& color) const = 0;
};

/** RGB-threshold classifier (the former ConeFusion::getConeColor). */
class RgbThresholdClassifier : public IColorClassifier {
public:
  uint8_t classify(const std_msgs::msg::ColorRGBA& color) const override {
    if (color.r < 0.1f && color.g < 0.1f && color.b > 0.9f) {
      return blueCone;
    } else if (color.r > 0.9f && color.g > 0.9f && color.b < 0.1f) {
      return yellowCone;
    } else if (color.r > 0.9f && color.g > 0.3f && color.b < 0.1f) {
      return orangeCone;        /* small orange (green ~0.31) */
    } else if (color.r > 0.9f && color.g > 0.6f && color.b < 0.1f) {
      return orangeBigCone;     /* big orange (green ~0.63) */
    }
    return 255;                 /* unknown -> ignored by ColorLogic voting */
  }
};

/** Always reports yellow (colour-agnostic mapping; the former is_colorblind path). */
class ColorblindClassifier : public IColorClassifier {
public:
  uint8_t classify(const std_msgs::msg::ColorRGBA& /*color*/) const override {
    return yellowCone;
  }
};

/**
 * Factory: build the concrete classifier once from the node parameters. New
 * strategies (e.g. a PerceptionLabelClassifier that trusts an upstream integer
 * label) plug in here without touching the adapter.
 */
inline std::unique_ptr<IColorClassifier> makeColorClassifier(bool is_colorblind) {
  if (is_colorblind)
    return std::make_unique<ColorblindClassifier>();
  return std::make_unique<RgbThresholdClassifier>();
}
