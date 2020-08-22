#include "header.h"

using namespace torch;

// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 100;

struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl(int kNoiseSize)
        : conv1(nn::ConvTranspose1dOptions(kNoiseSize, 512, 4)
            .bias(false)),
        batch_norm1(512),
        conv2(nn::ConvTranspose2dOptions(512, 256, 4)
            .stride(2)
            .padding(1)
            .bias(false)),
        batch_norm2(256),
        conv3(nn::ConvTranspose2dOptions(256, 128, 4)
            .stride(2)
            .padding(1)
            .bias(false)),
        batch_norm3(128),
        conv4(nn::ConvTranspose2dOptions(128, 64, 4)
            .stride(2)
            .padding(1)
            .bias(false)),
        batch_norm4(64),
        conv5(nn::ConvTranspose2dOptions(64, 3, 4)
            .stride(2)
            .padding(1)
            .bias(false))
    {
        // register_module() is needed if we want to use the parameters() method later on
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        register_module("batch_norm1", batch_norm1);
        register_module("batch_norm2", batch_norm2);
        register_module("batch_norm3", batch_norm3);
        register_module("batch_norm4", batch_norm4);
    }

    Tensor forward(Tensor x) {
        x = relu(batch_norm1(conv1(x)));
        x = relu(batch_norm2(conv2(x)));
        x = relu(batch_norm3(conv3(x)));
        x = relu(batch_norm4(conv4(x)));
        x = tanh(conv5(x));
        return x;
    }

    nn::ConvTranspose2d conv1, conv2, conv3, conv4, conv5;
    nn::BatchNorm2d batch_norm1, batch_norm2, batch_norm3, batch_norm4;
};



int main() {

	Tensor tensor1 = torch::rand({5,9});
	Tensor tensor2 = torch::rand({ 9,5 });
	Tensor tensor3 = torch::rand({ 4,3 }, at::kCUDA);
	std::cout << torch::mm(tensor1, tensor2) << std::endl;
	std::cout << "GPU Available: " << torch::cuda::is_available() << std::endl;
	std::cout << tensor3 << std::endl;
	std::cin.get();

}