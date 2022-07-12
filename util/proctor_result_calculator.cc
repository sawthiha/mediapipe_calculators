#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "proctor_result.h"

namespace mediapipe
{
    /**
     * @brief Proctor Result Calculator
     * 
     * OUTPUTS:
     *      RESULT - Proctoring Result <ProctorResult>
     * 
     * Example:
     * 
     * # Proctor Result Calculator
     *  node  {
     *      calculator: "ProctorResultCalculator"
     *      input_stream: "ALIGN:face_alignments"
     *      input_stream: "BLINK:face_blinks"
     *      input_stream: "ACTIVE:face_activity"
     *      input_stream: "MOVE:face_movement"
     *      output_stream: "RESULT:result"
     * 
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
        cc->Inputs().Tag("ALIGN").Set<std::map<std::string, double>>();
        cc->Inputs().Tag("BLINK").Set<std::map<std::string, double>>();
        cc->Inputs().Tag("ACTIVE").Set<double>();
        cc->Inputs().Tag("MOVE").Set<double>();
        cc->Outputs().Tag("RESULT").Set<ProctorResult>();

        return absl::OkStatus();
    }

    absl::Status ProctorResultCalculator::Open(CalculatorContext* cc)
    {
        return absl::OkStatus();
    }

    absl::Status ProctorResultCalculator::Process(CalculatorContext* cc)
    {
        ProctorResult result;
        auto blink = cc->Inputs().Tag("BLINK").Get<std::map<std::string, double>>();
        auto threshold = blink.at("threshold");
        result.is_left_eye_blinking = blink.at("left") < threshold;
        result.is_right_eye_blinking = blink.at("right") < threshold;
        
        auto alignment = cc->Inputs().Tag("ALIGN").Get<std::map<std::string, double>>();
        result.horizontal_align = alignment.at("horizontal_align");
        result.vertical_align   = alignment.at("vertical_align");
            
        result.facial_activity = cc->Inputs().Tag("ACTIVE").Get<double>();
        result.face_movement = cc->Inputs().Tag("MOVE").Get<double>();

        Packet packet = MakePacket<decltype(result)>(result).At(cc->InputTimestamp()); 
        cc->Outputs().Tag("RESULT").AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status ProctorResultCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

    typedef EndLoopCalculator<std::vector<ProctorResult>> EndLoopProctorResultVectorCalculator;
    REGISTER_CALCULATOR(EndLoopProctorResultVectorCalculator);

} // namespace mediapipe

