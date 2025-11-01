#include "object_detection_yoloV7.h"
#include <cstring>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#define SAMPLE_DATA_PATH "../sample_data"

static constexpr const char *ModelPath = SAMPLE_DATA_PATH "/yolov7_640x640.onnx";
static constexpr const bool _UseCuda = true;

int main()
{
    std::cout << "------- Runner of object_detection_yoloV7 -------" << "\n";

    // Setup environment and session options
    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::SessionOptions sessionOptions;
    OrtCUDAProviderOptions cuda_options;

    // Set number of threads
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(1);

    // Optimization will take time and memory during startup
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    if (_UseCuda)
    {
        try
        {                                                                           // Try to get CUDA provider and options
            cuda_options.device_id = 0;                                             // GPU_ID
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
            cuda_options.arena_extend_strategy = 0;
            // May cause data race in some condition
            cuda_options.do_copy_in_default_stream = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options
        }
        catch (const Ort::Exception &e)
        {
            std::cout << "ERROR: Unable to set CUDA provider: " << e.what() << "\n";
            std::cout << "Falling back to CPU execution provider.\n";
        }
    }

    // Create session
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator{};

    try
    {
        // Model path is const wchar_t*
        session = Ort::Session(env, ModelPath, sessionOptions);
    }
    catch (Ort::Exception oe)
    {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        return -1;
    }

    Ort::MemoryInfo memory_info{nullptr}; // Used to allocate memory for input
    try
    {
        memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
    catch (Ort::Exception oe)
    {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        return -1;
    }

    // Get input node info through ORT API
    size_t num_input_nodes = 0;
    std::vector<std::string> input_node_names{};  // Input node names
    std::vector<std::string> output_node_names{}; // Output node names

    std::vector<std::vector<int64_t>> input_node_dims; // Input node dimension.
    ONNXTensorElementDataType type;                    // Used to print input info
    Ort::TypeInfo *type_info;

    num_input_nodes = session.GetInputCount();

    input_node_names.reserve(num_input_nodes);
    output_node_names.reserve(1);

    std::cout << " ------- Input info ------- " << "\n";
    std::cout << "Input count: " << num_input_nodes << "\n";
    for (int i = 0; i < num_input_nodes; i++)
    {

        Ort::AllocatedStringPtr name_ptr = session.GetInputNameAllocated(i, allocator);
        input_node_names.emplace_back(name_ptr.get()); // Safe copy of input name

        type_info = new Ort::TypeInfo(session.GetInputTypeInfo(i));

        // Get tensor type and shape
        auto tensor_info = type_info->GetTensorTypeAndShapeInfo();
        type = tensor_info.GetElementType();
        input_node_dims.push_back(tensor_info.GetShape());

        // Print input shapes/dims
        printf("Input %d : name=%s\n", i, input_node_names.back().c_str()); // FIXME
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.back().size());
        for (int j = 0; j < input_node_dims.back().size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims.back()[j]);
        printf("Input %d : type=%d\n", i, type);

        delete (type_info);
    }

    output_node_names.push_back("output"); // Hardcoded output name for YOLOv7

    // Process input image
    std::vector<Ort::Value> inputTensor; // Onnxruntime allowed input

    // Load image
    cv::Mat image;
    image = cv::imread(SAMPLE_DATA_PATH "/horses.jpg", 1);

    cv::namedWindow("Loaded Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Loaded Image", image);
    cv::waitKey(0);

    // Preprocess: Resize + Normalize + BGR->RGB + HWC->CHW + Convert to float
    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(640, 640), (0, 0, 0), false, false);
    size_t input_tensor_size = blob.total();
    try
    {
        // Create input tensor object from data values
        inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float *)blob.data, input_tensor_size, input_node_dims[0].data(), input_node_dims[0].size()));
    }
    catch (Ort::Exception oe)
    {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        return -1;
    }

    // Run inference
    std::vector<Ort::Value> outputTensor;

    try
    {
        // Convert to const char* for Run
        std::vector<const char *> input_names_c;
        input_names_c.reserve(input_node_names.size());
        for (auto &s : input_node_names)
            input_names_c.push_back(s.c_str());

        std::vector<const char *> output_names_c;
        for (auto &s : output_node_names)
            output_names_c.push_back(s.c_str());

        // Try to run the session with input tensor data (1,3,640,640)
        outputTensor = session.Run(Ort::RunOptions{nullptr},
                                   input_names_c.data(),
                                   inputTensor.data(),
                                   inputTensor.size(),
                                   output_names_c.data(), 1);
    }
    catch (Ort::Exception oe)
    {
        std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        return -1;
    }

    // Get result from output tensor
    if (outputTensor.empty())
    {
        std::cout << "ERROR: Inference returned no output tensors.\n";
        return -1;
    }

    Ort::Value &ort_output = outputTensor.front();
    auto output_info = ort_output.GetTensorTypeAndShapeInfo();
    auto output_shape = output_info.GetShape();
    const float *output_data = ort_output.GetTensorData<float>();

    if (output_data == nullptr)
    {
        std::cout << "ERROR: Output tensor data is null.\n";
        return -1;
    }

    // Interpret output tensor
    bool channels_last = true;
    size_t num_detections = 0;
    size_t attributes_per_det = 0;

    // TODO: review implementation and fix plots
    if (output_shape.size() == 3)
    {
        // Interpret case: [1, N, M]
        size_t dim1 = static_cast<size_t>(output_shape[1]);
        size_t dim2 = static_cast<size_t>(output_shape[2]);

        // Check which dimension is number of detections and number of classes
        if (dim1 >= dim2)
        {
            channels_last = false;
            attributes_per_det = dim1;
            num_detections = dim2;
        }
        else
        {
            channels_last = true;
            num_detections = dim1;
            attributes_per_det = dim2;
        }
    }
    else if (output_shape.size() == 2)
    {
        // Interpret case: [N, M]
        size_t dim0 = static_cast<size_t>(output_shape[0]);
        size_t dim1 = static_cast<size_t>(output_shape[1]);
        
        if (dim0 >= dim1)
        {
            channels_last = false;
            attributes_per_det = dim0;
            num_detections = dim1;
        }
        else
        {
            channels_last = true;
            num_detections = dim0;
            attributes_per_det = dim1;
        }
    }
    else
    {
        // Interpret case: only one detection with all attributes
        attributes_per_det = 85;
        num_detections = output_info.GetElementCount() / attributes_per_det;
        channels_last = true;
    }

    if (attributes_per_det < 5 || num_detections == 0)
    {
        std::cout << "ERROR: Unexpected output tensor shape.\n";
        return -1;
    }

    const size_t num_classes = attributes_per_det > 5 ? attributes_per_det - 5 : 0;
    if (num_classes == 0)
    {
        std::cout << "ERROR: No class scores present in output tensor.\n";
        return -1;
    }

    const float x_factor = static_cast<float>(image.cols) / 640.0f;
    const float y_factor = static_cast<float>(image.rows) / 640.0f;

    const float confThreshold = 0.25f;
    const float scoreThreshold = 0.25f;
    const float nmsThreshold = 0.45f;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    class_ids.reserve(num_detections);
    confidences.reserve(num_detections);
    boxes.reserve(num_detections);

    // Check output for each detection
    for (size_t i = 0; i < num_detections; ++i)
    {
        float objectness = channels_last ? output_data[i * attributes_per_det + 4]
                                         : output_data[4 * num_detections + i];

        // If objectness score is less than threshold, discard detection
        if (objectness < confThreshold)
            continue;

        size_t best_class = 0;
        float best_class_score = channels_last ? output_data[i * attributes_per_det + 5]
                                               : output_data[5 * num_detections + i];

        // Check class scores
        for (size_t c = 1; c < num_classes; ++c)
        {
            float score = channels_last ? output_data[i * attributes_per_det + 5 + c]
                                        : output_data[(5 + c) * num_detections + i];
            if (score > best_class_score)
            {
                best_class_score = score;
                best_class = c;
            }
        }

        float score = objectness * best_class_score;
        if (score < scoreThreshold)
            continue;
        
        // Get bounding box corresponding to object
        float cx = channels_last ? output_data[i * attributes_per_det]
                                 : output_data[0 * num_detections + i];
        float cy = channels_last ? output_data[i * attributes_per_det + 1]
                                 : output_data[1 * num_detections + i];
        float width = channels_last ? output_data[i * attributes_per_det + 2]
                                    : output_data[2 * num_detections + i];
        float height = channels_last ? output_data[i * attributes_per_det + 3]
                                     : output_data[3 * num_detections + i];

        // Convert to top-left corner format for plot
        int left = static_cast<int>((cx - 0.5f * width) * x_factor);
        int top = static_cast<int>((cy - 0.5f * height) * y_factor);
        int box_width = static_cast<int>(width * x_factor);
        int box_height = static_cast<int>(height * y_factor);

        if (box_width <= 0 || box_height <= 0)
            continue;

        // Saturate box coordinates to image size for plot
        if (left < 0)
            left = 0;
        if (top < 0)
            top = 0;
        if (left + box_width > image.cols)
            box_width = image.cols - left;
        if (top + box_height > image.rows)
            box_height = image.rows - top;

        if (box_width <= 0 || box_height <= 0)
            continue;
        
        // Store data
        boxes.emplace_back(left, top, box_width, box_height); // Bounding box coordinates
        confidences.emplace_back(score); // Confidence score
        class_ids.emplace_back(static_cast<int>(best_class)); // Class ID
    }

    // Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, scoreThreshold, nmsThreshold, indices);

    if (indices.empty())
    {
        std::cout << "No detections above thresholds.\n";
    }
    else
    {   
        // Plot detections on image
        std::cout << "Detections kept: " << indices.size() << "\n";
        for (int idx : indices)
        {   
            // Add rectangle
            const cv::Rect &box = boxes[idx];
            cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

            std::string label = cv::format("cls %d %.2f", class_ids[idx], confidences[idx]);
            int text_y = box.y > 10 ? box.y - 5 : box.y + 15;
            
            // Add text label
            cv::putText(image, label, cv::Point(box.x, text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            std::cout << " - class: " << class_ids[idx] << ", score: " << confidences[idx]
                      << ", box: [" << box.x << ", " << box.y << ", " << box.width << ", " << box.height << "]\n";
        }

        cv::imshow("Loaded Image", image);
        cv::waitKey(0);
    }

    inputTensor.clear();

    return 0;
}
