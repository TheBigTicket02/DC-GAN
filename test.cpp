#include "header.h"

using namespace torch;

// Reshaped Image Size
const int64_t kImageSize = 64;

// The size of the noise vector fed to the generator.
const int64_t kLatentDim = 100;

// The batch size for training.
const int64_t kBatchSize = 64;

// Number of workers
const int64_t kNumOfWorkers = 4;

// Enforce ordering
const bool kEnforceOrder = false;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 30;

// Where to find the CSV with file locations.
const string kCsvFile = "../file_names.csv";

// After how many batches to create a new checkpoint periodically.
const int64_t kCheckpointEvery = 200;

// How many images to sample at every checkpoint.
const int64_t kNumberOfSamplesPerCheckpoint = 64;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

// Learning Rate
const double kLr = 2e-4;

// Beta1
const double kBeta1 = 0.5;

// Beta2
const double kBeta2 = 0.999;

struct DCGANGeneratorImpl : nn::Module {
    DCGANGeneratorImpl(int kLatentDim)
        : conv1(nn::ConvTranspose2dOptions(kLatentDim, 512, 4)
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

TORCH_MODULE(DCGANGenerator);

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

        csv.push_back(std::make_tuple("../" + name, stoi(label)));
    }
    return csv;
};

struct FaceDataset : torch::data::Dataset<FaceDataset>
{

    std::vector<std::tuple<std::string /*file location*/, int64_t /*label*/>> csv_;

    FaceDataset(std::string& file_names_csv)
        // Load csv file with file locations and labels.
        : csv_(ReadCsv(file_names_csv)) {

    };

    // Override the get method to load custom data.
    torch::data::Example<> get(size_t index) override {

        std::string file_location = std::get<0>(csv_[index]);
        int64_t label = std::get<1>(csv_[index]);

        // Load image with OpenCV.
        cv::Mat img = cv::imread(file_location);

        cv::resize(img, img, cv::Size(kImageSize,kImageSize));

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
    manual_seed(42);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    DCGANGenerator generator(kLatentDim);
    generator->to(device);

    nn::Sequential discriminator(
        // Layer 1
        nn::Conv2d(
            nn::Conv2dOptions(3, 64, 4).stride(2).padding(1).bias(false)),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 2
        nn::Conv2d(
            nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(128),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 3
        nn::Conv2d(
            nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(256),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 4
        nn::Conv2d(
            nn::Conv2dOptions(256, 512, 4).stride(2).padding(1).bias(false)),
        nn::BatchNorm2d(512),
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2)),
        // Layer 5
        nn::Conv2d(
            nn::Conv2dOptions(512, 1, 4).stride(1).padding(0).bias(false)),
        nn::Sigmoid());
    discriminator->to(device);

    std::string file_names_csv = kCsvFile;
    auto dataset = FaceDataset(file_names_csv)
        .map(data::transforms::Normalize<>(0.5, 0.5))
        .map(data::transforms::Stack<>());

    const int64_t batches_per_epoch =
        std::ceil(dataset.size().value() / static_cast<double>(kBatchSize));

    auto data_loader = data::make_data_loader<data::samplers::RandomSampler>(
        dataset,
        data::DataLoaderOptions().workers(kNumOfWorkers).batch_size(kBatchSize).enforce_ordering(kEnforceOrder));

    optim::Adam generator_optimizer(
        generator->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));
    optim::Adam discriminator_optimizer(
        discriminator->parameters(), optim::AdamOptions(kLr).betas(std::make_tuple(kBeta1, kBeta2)));


    int64_t checkpoint_counter = 1;
    for (int64_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
        int64_t batch_index = 0;
        for (auto& batch : *data_loader) {
            // Train discriminator with real images.
            discriminator->zero_grad();
            Tensor real_images = batch.data.to(device);
            Tensor real_labels =
                torch::empty(batch.data.size(0), device).uniform_(0.8, 1.0);
            Tensor real_output = discriminator->forward(real_images);
            Tensor d_loss_real =
                binary_cross_entropy(real_output, real_labels);
            d_loss_real.backward();

            // Train discriminator with fake images.
            Tensor noise =
                torch::randn({ batch.data.size(0), kLatentDim, 1, 1 }, device);
            Tensor fake_images = generator->forward(noise);
            Tensor fake_labels = torch::zeros(batch.data.size(0), device);
            Tensor fake_output = discriminator->forward(fake_images.detach());
            Tensor d_loss_fake =
                torch::binary_cross_entropy(fake_output, fake_labels);
            d_loss_fake.backward();

            Tensor d_loss = d_loss_real + d_loss_fake;
            discriminator_optimizer.step();

            // Train generator.
            generator->zero_grad();
            fake_labels.fill_(1);
            fake_output = discriminator->forward(fake_images);
            Tensor g_loss =
                torch::binary_cross_entropy(fake_output, fake_labels);
            g_loss.backward();
            generator_optimizer.step();
            batch_index++;
            if (batch_index % kLogInterval == 0) {
                std::printf(
                    "\r[%2ld/%2ld][%3ld/%3ld] D_loss: %.4f | G_loss: %.4f\n",
                    epoch,
                    kNumberOfEpochs,
                    batch_index,
                    batches_per_epoch,
                    d_loss.item<float>(),
                    g_loss.item<float>());
            }
            if (batch_index % kCheckpointEvery == 0) {
                // Checkpoint the model and optimizer state.
                torch::save(generator, "generator-checkpoint.pt");
                torch::save(generator_optimizer, "generator-optimizer-checkpoint.pt");
                torch::save(discriminator, "discriminator-checkpoint.pt");
                torch::save(
                    discriminator_optimizer, "discriminator-optimizer-checkpoint.pt");
                // Sample the generator and save the images.
                Tensor samples = generator->forward(torch::randn(
                    { kNumberOfSamplesPerCheckpoint, kLatentDim, 1, 1 }, device));
                torch::save(
                    (samples + 1.0) / 2.0,
                    torch::str("dcgan-sample-", checkpoint_counter, ".pt"));
                std::cout << "\n-> checkpoint " << ++checkpoint_counter << '\n';
            }
        }
    }

    std::cout << "Training complete!" << std::endl;
}