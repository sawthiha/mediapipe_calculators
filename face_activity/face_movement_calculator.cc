#include <vector>
#include <map>
#include <optional>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe
{
    /**
     * @brief Detect face position changes on screen
     * 
     * INPUTS:
     *      0 - Landmarks (NormalizedLandmarkList)
     * OUTPUTS:
     *      0 - Face Position Delta (double)
     * 
     * Example:
     * 
     * node {
     *   calculator: "FaceMovementCalculator"
     *   input_stream: "face_landmarks"
     *   output_stream: "face_movement"
     * }
     * 
     */
    class FaceMovementCalculator: public CalculatorBase
    {
    private:
        cv::Vec3f m_prev_vec;

    public:
        FaceMovementCalculator() = default;
        ~FaceMovementCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(FaceMovementCalculator);

    absl::Status FaceMovementCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Index(0).Set<NormalizedLandmarkList>();
        cc->Outputs().Index(0).Set<double>();
        return absl::OkStatus();
    }

    absl::Status FaceMovementCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }

    absl::Status FaceMovementCalculator::Process(CalculatorContext* cc)
    {
        auto landmarks = cc->Inputs().Index(0).Get<NormalizedLandmarkList>();
        auto cur_landmark = landmarks.landmark(0);
        cv::Vec3f cur_vec (cur_landmark.x(), cur_landmark.y(), cur_landmark.z());
        auto delta = cv::norm(cur_vec - m_prev_vec, cv::NORM_L2);
        m_prev_vec = cur_vec;
            
        Packet packet = MakePacket<decltype(delta)>(delta).At(cc->InputTimestamp());
        cc->Outputs().Index(0).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status FaceMovementCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
