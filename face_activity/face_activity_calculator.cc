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
     * @brief Detect facial activity changes
     * 
     * INPUTS:
     *      0 - Standardized Landmarks (std::vector<NormalizedLandmarkList>)
     * OUTPUTS:
     *      0 - Facial Activity Deltas (std::vector<double> )
     * 
     * Example:
     * 
     * node {
     *   calculator: "FaceActivityCalculator"
     *   input_stream: "multi_face_std_landmarks"
     *   output_stream: "multi_face_activities"
     * }
     * 
     */
    class FaceMovementCalculator: public CalculatorBase
    {
    private:
        cv::Mat m_prev_landmark_mat;

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
        std::vector<double > multi_face_activities;
        if (!cc->Inputs().Index(0).IsEmpty())
        {
            const auto& multi_face_landmarks = cc->Inputs().Index(0).Get<std::vector<NormalizedLandmarkList> >();
            for(auto&& landmarks: multi_face_landmarks)
            {
                cv::Mat cur_landmark_mat(landmarks.landmark_size(), 3, CV_64FC1);

                for (int i = 0; i < landmarks.landmark_size(); ++i) {
                    cur_landmark_mat.at<double>(i, 0) = landmarks.landmark(i).x();
                    cur_landmark_mat.at<double>(i, 1) = landmarks.landmark(i).y();
                    cur_landmark_mat.at<double>(i, 2) = landmarks.landmark(i).z();
                }
                multi_face_activities.push_back(cv::norm(cur_landmark_mat - m_prev_landmark_mat, cv::NORM_L2));
                m_prev_landmark_mat = cur_landmark_mat;
            }
        }
            
        Packet packet = MakePacket<decltype(multi_face_activities)>(multi_face_activities).At(cc->InputTimestamp());
        cc->Outputs().Index(0).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status FaceMovementCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

    typedef BeginLoopCalculator<std::vector<double > >
        BeginLoopFacialActivityVectorCalculator;
    // Register the begin loop calculator
    REGISTER_CALCULATOR(BeginLoopFacialActivityVectorCalculator);

} // namespace mediapipe
