#include "header.h"

using namespace torch;

// The size of the noise vector fed to the generator.
const int64_t kNoiseSize = 100;

struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl(int kNoiseSize)
        : conv1(nn::ConvTranspose2dOptions(kNoiseSize, 512, 4)
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

auto ReadCsv(std::string& location) {
    std::fstream in(location, std::ios::in);
    std::string line;
    std::string name;
    std::string label;
    std::vector<std::tuple<std::string, int64_t>> csv;

    while (getline(in, line)) {
        std::stringstream s(line);
        getline(s, name, ',');
        getline(s, label, ',');

        csv.push_back(std::make_tuple(name, stoi(label)));
    }
    return csv;
};

struct Dataset : torch::data::Dataset<Dataset>
{

    std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;

    Dataset(std::string& file_names_csv)
        // Load csv file with file locations and labels.
        : csv_(ReadCsv(file_names_csv)) {

    };

    // Override the get method to load custom data.
    torch::data::Example<> get(size_t index) override {

        std::string file_location = std::get<0>(csv_[index]);
        int64_t label = std::get<1>(csv_[label]);

        // Load image with OpenCV.
        cv::Mat img = cv::imread(file_location);

        // Convert the image and label to a tensor.
        // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
        // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
        // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
        Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte).clone();
        img_tensor = img_tensor.permute({ 2, 0, 1 }); // convert to CxHxW
        Tensor label_tensor = torch::full({ 1 }, label);

        return { img_tensor, label_tensor };
    };

    // Override the size method to infer the size of the data set.
    torch::optional<size_t> size() const override {

        return csv_.size();
    };
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