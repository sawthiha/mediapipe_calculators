#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "proctor_result.h"

namespace mediapipe
{
    /**
     * @brief Proctor Result Calculator
     * 
     * OUTPUTS:
     *      RESULT - Proctoring Result <std::vector<ProctorResult>>
     * 
     * Example:
     * 
     * # Proctor Result Calculator
     *  node  {
     *      calculator: "ProctorResultCalculator"
     *      input_stream: "TICK:sync_stream"
     *      input_stream: "ALIGN:multi_face_alignments"
     *      input_stream: "BLINK:multi_face_blinks"
     *      input_stream: "ACTIVE:multi_face_activities"
     *      input_stream: "MOVE:multi_face_movements"
     *      output_stream: "RESULT:result"
     *  }
     * 
     */
    class ProctorResultCalculator: public CalculatorBase
    {

    public:
        ProctorResultCalculator() = default;
        ~ProctorResultCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(ProctorResultCalculator);

    absl::Status ProctorResultCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag("TICK").SetAny();
        cc->Inputs().Tag("ALIGN").Set<std::vector<std::map<std::string, double> >>();
        cc->Inputs().Tag("BLINK").Set<std::vector<std::map<std::string, double> >>();
        cc->Inputs().Tag("ACTIVE").Set<std::vector<double>>();
        cc->Inputs().Tag("MOVE").Set<std::vector<double>>();
        cc->Outputs().Tag("RESULT").Set<std::vector<ProctorResult>>();
        return absl::OkStatus();
    }

    absl::Status ProctorResultCalculator::Open(CalculatorContext* cc)
    {
        return absl::OkStatus();
    }

    absl::Status ProctorResultCalculator::Process(CalculatorContext* cc)
    {

        std::vector<ProctorResult> results;
        auto alignments = cc->Inputs().Tag("ALIGN").Get<std::vector<std::map<std::string, double> >>();
        auto blinks = cc->Inputs().Tag("BLINK").Get<std::vector<std::map<std::string, double> >>();
        auto activities = cc->Inputs().Tag("ACTIVE").Get<std::vector<double>>();
        auto movements = cc->Inputs().Tag("MOVE").Get<std::vector<double>>();

        for (size_t i = 0; i < alignments.size(); i++)
        {
            ProctorResult result;
            auto blink = blinks.at(i);
            auto threshold = blink.at("threshold");
            result.is_left_eye_blinking = blink.at("left") < threshold;
            result.is_right_eye_blinking = blink.at("right") < threshold;
            
            auto alignment = multi_face_alignments.at(i);
            result.horizontal_align = alignment.at("horizontal_align");
            result.vertical_align   = alignment.at("vertical_align");
            
            result.facial_activity = activities.at(i);
            result.face_movement = movements.at(i);

            results.push_back(std::move(result));
        }

        Packet packet = MakePacket<decltype(results)>(results).At(cc->InputTimestamp());
        cc->Outputs().Tag("RESULT").AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status ProctorResultCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe

