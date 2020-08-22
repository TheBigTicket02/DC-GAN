#include "header.h"

using namespace torch;


int main() {
	Tensor tensor1 = torch::rand({5,9});
	Tensor tensor2 = torch::rand({ 9,5 });
	Tensor tensor3 = torch::rand({ 4,2 }, at::kCUDA);
	std::cout << torch::mm(tensor1, tensor2) << std::endl;
	std::cout << "GPU Available: " << torch::cuda::is_available() << std::endl;
	std::cout << tensor3 << std::endl; 
	std::cin.get();

}