#include <vector>
#include <map>
#include <optional>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/calculators/core/begin_loop_calculator.h"

namespace mediapipe
{
    /**
     * @brief Detect various face activities such as position on screen and facial movements
     * 
     * INPUTS:
     *      0 - Landmarks (std::vector<NormalizedLandmarkList>)
     * OUTPUTS:
     *      0 - Face Position Deltas (std::vector<double> )
     * 
     * Example:
     * 
     * node {
     *   calculator: "FaceMovementCalculator"
     *   input_stream: "multi_face_landmarks"
     *   output_stream: "multi_face_movements"
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
        cc->Inputs().Index(0).Set<std::vector<NormalizedLandmarkList> >();
        cc->Outputs().Index(0).Set<std::vector<double> >();
        return absl::OkStatus();
    }

    absl::Status FaceMovementCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }

    absl::Status FaceMovementCalculator::Process(CalculatorContext* cc)
    {
        std::vector<double > multi_face_movements;
        if (!cc->Inputs().Index(0).IsEmpty())
        {
            const auto& multi_face_landmarks = cc->Inputs().Index(0).Get<std::vector<NormalizedLandmarkList> >();
            for(auto&& landmarks: multi_face_landmarks)
            {
                auto cur_landmark = landmarks.landmark(0);
                cv::Vec3f cur_vec (cur_landmark.x(), cur_landmark.y(), cur_landmark.z());
                multi_face_movements.push_back(cv::norm(cur_vec - m_prev_vec, cv::NORM_L2));
                m_prev_vec = cur_vec;
            }
        }
            
        Packet packet = MakePacket<decltype(multi_face_movements)>(multi_face_movements).At(cc->InputTimestamp());
        cc->Outputs().Index(0).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status FaceMovementCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

    typedef BeginLoopCalculator<std::vector<double > >
        BeginLoopMovementVectorCalculator;
    // Register the begin loop calculator
    REGISTER_CALCULATOR(BeginLoopMovementVectorCalculator);

} // namespace mediapipe
