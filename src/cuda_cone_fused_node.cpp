#include <cuda_cone_fused/cuda_cone_fused.hpp>
#include <unistd.h>

/* No SIGINT handler here on purpose. rclcpp::init() installs one already
   (InitOptions::shutdown_on_signal defaults to true), and its async part does
   nothing but sem_post -- the real teardown runs on rclcpp's own
   deferred_signal_handler thread, then spin() returns and the shutdown() below
   runs on the main thread.

   A hand-rolled handler calling rclcpp::shutdown() directly bypasses that.
   rclcpp chains into any handler installed before init(), so the call lands in
   async signal context, on whatever thread the kernel interrupted, and tears
   down the rmw participant from there. If that thread already held a FastDDS
   lock it deadlocks against itself and every other thread piles up behind it:
   the process survives SIGINT forever, holding its CUDA context, invisible to
   the ROS graph because its discovery sockets are never drained again. */

int main(int argc, char* argv[])
{
  /* node initialization */
  rclcpp::init(argc, argv);

  auto node = std::make_shared<ConeFusion>();

  /* MultiThreaded, with exactly TWO threads, because the node has exactly two
     callback groups: the default one holding every subscription (cones, odom,
     race_status -- still MutuallyExclusive, so still serialised against each
     other exactly as under rclcpp::spin) and the TF republish timer's own
     group. Two threads is what it takes for the timer to run while an EKF
     update is in progress, and one more than that would only float across the
     cores as_demo reserves for the rest of the stack. */
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), /*number_of_threads=*/2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();

  return 0;

}
