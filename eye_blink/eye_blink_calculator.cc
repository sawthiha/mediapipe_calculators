#include <map>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"

namespace mediapipe
{

    /**
     * @brief Detect eye blinks from Standardized Landmarks
     * 
     * INPUTS:
     *      0 - Standardized Landmarks (NormalizedLandmarkList)
     * OUTPUTS:
     *      0 - Eye Blink data (std::map<std::string, double>)
     *      {
     *          'left': double, lower value means eye is closing
     *          'right': double, lower value means eye is closing
     *          'threshold': double, a threshold value for detection, e.g. left eye is blinking if 'left' < 'threshold'
     *      }
     * 
     * Example:
     * 
     * node {
     *   calculator: "EyeBlinkCalculator"
     *   input_stream: "face_std_landmarks"
     *   output_stream: "face_blinks"
     * }
     * 
     */
    class EyeBlinkCalculator: public CalculatorBase
    {
    public:
        EyeBlinkCalculator() = default;
        ~EyeBlinkCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;

    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(EyeBlinkCalculator);

    absl::Status EyeBlinkCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Index(0).Set<NormalizedLandmarkList>();
        cc->Outputs().Index(0).Set<std::map<std::string, double>>();
        return absl::OkStatus();
    }

    absl::Status EyeBlinkCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }

    absl::Status EyeBlinkCalculator::Process(CalculatorContext* cc)
    {
        auto landmarks = cc->Inputs().Index(0).Get<NormalizedLandmarkList>();
        std::map<std::string, double> blink_map;
        
        // Right Eye
        cv::Vec3d ur_el { landmarks.landmark(386).x(), landmarks.landmark(386).y() };
        cv::Vec3d lr_el { landmarks.landmark(374).x(), landmarks.landmark(374).y() };

        // Left Eye
        cv::Vec3d ul_el { landmarks.landmark(159).x(), landmarks.landmark(159).y() };
        cv::Vec3d ll_el { landmarks.landmark(145).x(), landmarks.landmark(145).y() };

        auto r_dist = cv::norm(ur_el - lr_el, cv::NORM_L2);
        auto l_dist = cv::norm(ul_el - ll_el, cv::NORM_L2);

        blink_map["left"] = l_dist;
        blink_map["right"] = r_dist;
        blink_map["threshold"] = landmarks.landmark(1).x() * 0.0308 + landmarks.landmark(1).y() * 0.0803 + 0.1476;

        Packet packet = MakePacket<decltype(blink_map)>(blink_map).At(cc->InputTimestamp());
        cc->Outputs().Index(0).AddPacket(packet);

        return absl::OkStatus();
    } // Process

    absl::Status EyeBlinkCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
