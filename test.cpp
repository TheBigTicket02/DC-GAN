#include <torch/torch.h>
#include <iostream>

using namespace std;
using namespace torch;


int main() {
	Tensor tensor1 = torch::rand({5,9});
	Tensor tensor2 = torch::rand({ 9,2 });
	Tensor tensor3 = torch::rand({ 3,3 }, at::kCUDA);
	cout << torch::mm(tensor1, tensor2) << endl;
	cout << "GPU Available: " << torch::cuda::is_available() << endl;
	cout << tensor3 << endl; 
	cin.get();

}