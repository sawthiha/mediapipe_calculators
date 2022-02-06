#include <vector>
#include <map>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/calculators/core/begin_loop_calculator.h"

namespace mediapipe
{
    /**
     * @brief Detect face alignments from Standardized Landmarks
     * 
     * INPUTS:
     *      0 - Standardized Landmarks (std::vector<NormalizedLandmarkList>)
     * OUTPUTS:
     *      0 - Face Alignment data (std::vector<std::map<std::string, double> >)
     *      {
     *          "horizontal_align": 0.0 being neutral, + being right, - being left
     *          "vertical_align":   0.0 being neutral, + being down,  - being up
     *      }
     * 
     * Example:
     * 
     * node {
     *   calculator: "FaceAlignmentCalculator"
     *   input_stream: "multi_face_std_landmarks"
     *   output_stream: "multi_face_alignments"
     * }
     * 
     */
    class FaceAlignmentCalculator: public CalculatorBase
    {
    public:
        FaceAlignmentCalculator() = default;
        ~FaceAlignmentCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;

    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(FaceAlignmentCalculator);

    absl::Status FaceAlignmentCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Index(0).Set<std::vector<NormalizedLandmarkList> >();
        cc->Outputs().Index(0).Set<std::vector<std::map<std::string, double> > >();
        return absl::OkStatus();
    }

    absl::Status FaceAlignmentCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }

    absl::Status FaceAlignmentCalculator::Process(CalculatorContext* cc)
    {
        std::vector<std::map<std::string, double> > multi_face_alignments;
        if (!cc->Inputs().Index(0).IsEmpty())
        {
            const auto& multi_face_landmarks = cc->Inputs().Index(0).Get<std::vector<NormalizedLandmarkList> >();
            for(auto&& landmarks: multi_face_landmarks)
            {
                std::map<std::string, double> alignment_map;
                alignment_map["horizontal_align"]   = landmarks.landmark(1).x();
                alignment_map["vertical_align"]     = landmarks.landmark(1).y();
                multi_face_alignments.push_back(alignment_map);
            }
        }
            
        Packet packet = MakePacket<decltype(multi_face_alignments)>(multi_face_alignments).At(cc->InputTimestamp());
        cc->Outputs().Index(0).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status FaceAlignmentCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

    typedef BeginLoopCalculator<std::vector<std::map<std::string, double> > >
        BeginLoopAlignmentVectorCalculator;
    // Register the begin loop calculator
    REGISTER_CALCULATOR(BeginLoopAlignmentVectorCalculator);

} // namespace mediapipe
