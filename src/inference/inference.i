//*************************************************************************
// Wrapper-safe inference API.
//*************************************************************************

namespace ptafdeploy
{
namespace inference
{

#include <inference/inference_common.h>
#include <inference/inference_manager.h>
#include <inference/inference_matlab_adapters.h>
#include <inference/model_facade.h>
#include <inference/task_adapters.h>
#include <inference/task_value_types.h>

enum class EInferenceBackend {
    auto_backend,
    onnxruntime,
    tensorrt_engine
};

enum class EModelArtifact {
    auto_artifact,
    onnx,
    tensorrt_engine
};

enum class EExecutionTarget {
    cpu,
    cuda,
    tensorrt
};

enum class EModelRole {
    raw_tensor,
    centroiding,
    object_detection,
    feature_matching,
    tracking,
    optical_flow,
    custom
};

enum class EBoundingBoxEncoding {
    center_xywh,
    corners_xyxy
};

class STensorInfo
{
    STensorInfo();

    string name;
    string dtype;
    std::vector<int64_t> shape;
    string location;
};

class SFloatTensor
{
    SFloatTensor();
    SFloatTensor(string tensor_name,
                 std::vector<int64_t> tensor_shape,
                 std::vector<float> tensor_values);

    string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

class SRuntimeConfig
{
    SRuntimeConfig();

    ptafdeploy::inference::EInferenceBackend GetBackend() const;
    void SetBackend(ptafdeploy::inference::EInferenceBackend value);
    ptafdeploy::inference::EModelArtifact GetArtifact() const;
    void SetArtifact(ptafdeploy::inference::EModelArtifact value);
    bool GetAllowFallback() const;
    void SetAllowFallback(bool value);
    int GetDeviceId() const;
    void SetDeviceId(int value);
    int GetIntraOpNumThreads() const;
    int GetInterOpNumThreads() const;
    void SetThreadCounts(int intra_op_threads,
                         int inter_op_threads);
    bool GetEnableProfiling() const;
    void SetEnableProfiling(bool value);
    string GetLogId() const;
    void SetLogId(string value);
    int GetTensorRtOptimizationProfileIndex() const;
    void SetTensorRtOptimizationProfileIndex(int value);
    void ClearExecutionTargetPriority();
    void AddExecutionTarget(ptafdeploy::inference::EExecutionTarget target);
    void UseCpuOnly();
    void UseCudaWithCpuFallback();
    void UseTensorRtWithCudaFallback();
};

class SPoint2D
{
    SPoint2D();

    float x;
    float y;
};

class SSize2D
{
    SSize2D();

    float width;
    float height;
};

class SBoundingBox2D
{
    SBoundingBox2D();

    ptafdeploy::inference::SPoint2D center;
    ptafdeploy::inference::SSize2D size;
};

class SFeature2D
{
    SFeature2D();

    ptafdeploy::inference::SPoint2D position;
    float score;
};

class SClassScore
{
    SClassScore();

    int64_t class_id;
    float score;
};

class SDetection2D
{
    SDetection2D();

    ptafdeploy::inference::SBoundingBox2D bounds;
    ptafdeploy::inference::SClassScore classification;
};

class SFeatureRowSchema
{
    SFeatureRowSchema();

    int64_t attribute_axis;
    size_t x_index;
    size_t y_index;
    int64_t score_index;
};

class SDetectionRowSchema
{
    SDetectionRowSchema();

    int64_t attribute_axis;
    ptafdeploy::inference::EBoundingBoxEncoding box_encoding;
    size_t box_coordinate_0_index;
    size_t box_coordinate_1_index;
    size_t box_coordinate_2_index;
    size_t box_coordinate_3_index;
    int64_t objectness_index;
    size_t first_class_score_index;
    size_t class_score_count;
};

class CInferenceManager
{
    CInferenceManager();
    CInferenceManager(string model_path);

    void LoadModel(string model_path);
    void LoadModelWithRuntimeConfig(string model_path,
                                    ptafdeploy::inference::SRuntimeConfig runtime_config);
    void LoadModelWithThreadCounts(string model_path,
                                   int intra_op_num_threads,
                                   int inter_op_num_threads);

    string GetBackendDetail() const;
    size_t GetNumInputs() const;
    size_t GetNumOutputs() const;
    ptafdeploy::inference::STensorInfo GetInputInfo(size_t index) const;
    ptafdeploy::inference::STensorInfo GetOutputInfo(size_t index) const;

    ptafdeploy::inference::SFloatTensor InferSingleFloatTensor(
        const ptafdeploy::inference::SFloatTensor& input) const;
    std::vector<float> InferSingleFloatInput(const std::vector<float>& values,
                                             const std::vector<int64_t>& shape) const;
    std::vector<ptafdeploy::inference::SFloatTensor> InferFloatTensors(
        const std::vector<ptafdeploy::inference::SFloatTensor>& inputs) const;
};

class SModelRoleConfig
{
    SModelRoleConfig();
    SModelRoleConfig(ptafdeploy::inference::EModelRole model_role);
    SModelRoleConfig(ptafdeploy::inference::EModelRole model_role,
                     string preprocessing_pipeline,
                     string postprocessing_pipeline);

    ptafdeploy::inference::EModelRole role;
    string preprocessing;
    string postprocessing;
};

class SModelContract
{
    SModelContract();

    string config_path;
    string artifact_path;
    string role;
    string preprocessing;
    string postprocessing;
    ptafdeploy::inference::SRuntimeConfig runtime;
    string backend_detail;
    std::vector<ptafdeploy::inference::STensorInfo> inputs;
    std::vector<ptafdeploy::inference::STensorInfo> outputs;
};

class CModelFacade
{
    CModelFacade();
    CModelFacade(string model_path);
    CModelFacade(string model_path,
                 ptafdeploy::inference::EModelRole role);
    CModelFacade(string model_path,
                 ptafdeploy::inference::SModelRoleConfig role_config);

    void LoadModel(string model_path);
    void LoadModelWithRuntimeConfig(string model_path,
                                    ptafdeploy::inference::SRuntimeConfig runtime_config);
    void LoadModelWithRole(string model_path,
                           ptafdeploy::inference::EModelRole role);
    void LoadModelWithRoleAndRuntimeConfig(string model_path,
                                           ptafdeploy::inference::EModelRole role,
                                           ptafdeploy::inference::SRuntimeConfig runtime_config);
    void LoadModelWithRoleConfig(string model_path,
                                 ptafdeploy::inference::SModelRoleConfig role_config);
    void LoadModelWithRoleConfigAndRuntimeConfig(
        string model_path,
        ptafdeploy::inference::SModelRoleConfig role_config,
        ptafdeploy::inference::SRuntimeConfig runtime_config);
    void LoadModelConfig(string config_path);
    void LoadModelConfigWithRuntimeConfig(string config_path,
                                          ptafdeploy::inference::SRuntimeConfig runtime_config);

    ptafdeploy::inference::SModelContract GetContract() const;
    string GetRole() const;
    string GetPreprocessing() const;
    string GetPostprocessing() const;
    string GetBackendDetail() const;
    size_t GetNumInputs() const;
    size_t GetNumOutputs() const;
    ptafdeploy::inference::STensorInfo GetInputInfo(size_t index) const;
    ptafdeploy::inference::STensorInfo GetOutputInfo(size_t index) const;

    ptafdeploy::inference::SFloatTensor InferSingleFloatTensor(
        const ptafdeploy::inference::SFloatTensor& input) const;
    std::vector<float> InferSingleFloatInput(const std::vector<float>& values,
                                             const std::vector<int64_t>& shape) const;
    std::vector<ptafdeploy::inference::SFloatTensor> InferFloatTensors(
        const std::vector<ptafdeploy::inference::SFloatTensor>& inputs) const;

    static std::vector<string> GetRecognizedRoles();
};

ptafdeploy::inference::SFloatTensor MakeNchwFloatTensorFromHwcFloat(
    string tensor_name,
    const std::vector<float>& hwc_values,
    size_t height,
    size_t width,
    size_t channels,
    float scale,
    bool swap_rb);
ptafdeploy::inference::SBoundingBox2D MakeBoundingBox2DFromCenterSize(
    float center_x,
    float center_y,
    float width,
    float height);
ptafdeploy::inference::SBoundingBox2D MakeBoundingBox2DFromXyxy(
    float min_x,
    float min_y,
    float max_x,
    float max_y);
std::vector<float> GetBoundingBox2DXyxy(
    const ptafdeploy::inference::SBoundingBox2D& box);
std::vector<ptafdeploy::inference::SFeature2D> DecodeFeatureRows(
    const ptafdeploy::inference::SFloatTensor& output,
    const ptafdeploy::inference::SFeatureRowSchema& schema);
std::vector<ptafdeploy::inference::SDetection2D> DecodeDetectionRows(
    const ptafdeploy::inference::SFloatTensor& output,
    const ptafdeploy::inference::SDetectionRowSchema& schema,
    float score_threshold,
    size_t max_detections);

gtsam::Vector GetInputShapeVector(ptafdeploy::inference::CInferenceManager* manager,
                                  size_t index);
gtsam::Vector GetOutputShapeVector(ptafdeploy::inference::CInferenceManager* manager,
                                   size_t index);
gtsam::Vector InferSingleFloatInputVector(ptafdeploy::inference::CInferenceManager* manager,
                                          const gtsam::Vector& values,
                                          const gtsam::Vector& shape);
gtsam::Vector GetModelInputShapeVector(ptafdeploy::inference::CModelFacade* model,
                                       size_t index);
gtsam::Vector GetModelOutputShapeVector(ptafdeploy::inference::CModelFacade* model,
                                        size_t index);
gtsam::Vector InferModelSingleFloatInputVector(ptafdeploy::inference::CModelFacade* model,
                                               const gtsam::Vector& values,
                                               const gtsam::Vector& shape);
ptafdeploy::inference::SFloatTensor MakeFloatTensorFromVector(
    string tensor_name,
    const gtsam::Vector& values,
    const gtsam::Vector& shape);
gtsam::Vector GetFloatTensorShapeVector(const ptafdeploy::inference::SFloatTensor& tensor);
gtsam::Vector GetFloatTensorValuesVector(const ptafdeploy::inference::SFloatTensor& tensor);
ptafdeploy::inference::SFloatTensor MakeNchwFloatTensorFromHwcVector(
    string tensor_name,
    const gtsam::Vector& hwc_values,
    size_t height,
    size_t width,
    size_t channels,
    double scale,
    bool swap_rb);
gtsam::Matrix DecodeDetectionRowsMatrix(
    const ptafdeploy::inference::SFloatTensor& output,
    const ptafdeploy::inference::SDetectionRowSchema& schema,
    double score_threshold,
    size_t max_detections);

} // namespace inference
} // namespace ptafdeploy
