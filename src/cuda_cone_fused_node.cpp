#include <cuda_cone_fused/cuda_cone_fused.hpp>
#include <unistd.h>

void handleSignal(int signal) {
    if (signal == SIGINT) {
        std::cout << "Received SIGINT. Killing cuda_cone_fused process.\n";
        rclcpp::shutdown();
    }
}


int main(int argc, char* argv[])
{
  signal(SIGINT, handleSignal);
  /* node initialization */
  rclcpp::init(argc, argv);

  auto node = std::make_shared<ConeFusion>();

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;

}