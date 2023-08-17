#include <vector>
#include <map>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe
{
    /**
     * @brief Detect face orientations from Standardized Landmarks
     * 
     * INPUTS:
     *      0 - Standardized Landmarks (NormalizedLandmarkList)
     * OUTPUTS:
     *      0 - Face orientation data (std::map<std::string, double>)
     *      {
     *          "horizontal_align": 0.0 being neutral, + being right, - being left
     *          "vertical_align":   0.0 being neutral, + being down,  - being up
     *      }
     * 
     * Example:
     * 
     * node {
     *   calculator: "FaceOrientationCalculator"
     *   input_stream: "face_std_landmarks"
     *   output_stream: "face_orientations"
     * }
     * 
     */
    class FaceOrientationCalculator: public CalculatorBase
    {
    public:
        FaceOrientationCalculator() = default;
        ~FaceOrientationCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;

    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(FaceOrientationCalculator);

    absl::Status FaceOrientationCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Index(0).Set<NormalizedLandmarkList>();
        cc->Outputs().Index(0).Set<std::map<std::string, double>>();
        return absl::OkStatus();
    }

    absl::Status FaceOrientationCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }

    absl::Status FaceOrientationCalculator::Process(CalculatorContext* cc)
    {
        auto landmarks = cc->Inputs().Index(0).Get<NormalizedLandmarkList>();
        std::map<std::string, double> orientation_map;
        orientation_map["horizontal_align"]   = landmarks.landmark(1).x();
        orientation_map["vertical_align"]     = landmarks.landmark(1).y();
            
        Packet packet = MakePacket<decltype(orientation_map)>(orientation_map).At(cc->InputTimestamp());
        cc->Outputs().Index(0).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status FaceOrientationCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
