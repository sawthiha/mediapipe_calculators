#include <vector>
#include <map>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe
{

    namespace
    {
        constexpr char kBlinkStreamTag[]  = "BLINK";
        constexpr char kRenderDataStreamTag[] = "RENDER";
    } // namespace

    /**
     * @brief Annotate Detected Eye Blink
     * 
     * INPUTS:
     *      BLINK - Blinks (std::vector<std::map<std::string, double> >)
     * OUTPUTS:
     *      RENDER - Render Data to be render by OverlayRenderer (RenderData)
     * 
     * Example:
     * 
     * node {
     *   calculator: "EyeBlinkToRenderDataCalculator"
     *   input_stream: "BLINK:multi_face_blinks"
     *   output_stream: "RENDER:blink_render_data"
     * }
     * 
     */
    class EyeBlinkToRenderDataCalculator: public CalculatorBase
    {
    private:
        void AnnotateBlink(RenderData& render_data, std::string blink, double left_pos);

    public:
        EyeBlinkToRenderDataCalculator() = default;
        ~EyeBlinkToRenderDataCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;

    };
    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(EyeBlinkToRenderDataCalculator);

    absl::Status EyeBlinkToRenderDataCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag(kBlinkStreamTag).Set<std::vector<std::map<std::string, double> > >();
        cc->Outputs().Tag(kRenderDataStreamTag).Set<RenderData>();
        return absl::OkStatus();
    }

    absl::Status EyeBlinkToRenderDataCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }


    void EyeBlinkToRenderDataCalculator::AnnotateBlink(RenderData& render_data, std::string blink, double left_pos)
    {
        auto annotation = render_data.add_render_annotations();
        if(blink == "")
        {
            annotation->mutable_color()->set_r(0);
            annotation->mutable_color()->set_g(255);
            annotation->mutable_color()->set_b(0);
        }else
        {
            annotation->mutable_color()->set_r(255);
            annotation->mutable_color()->set_g(0);
            annotation->mutable_color()->set_b(0);
        }
        annotation->set_thickness(5);
        auto text = annotation->mutable_text();
        text->set_font_height(0.05);
        text->set_font_face(0);
        text->set_display_text(blink);
        text->set_normalized(true);
        text->set_left(left_pos);
        text->set_baseline(0.25);
    } // AnnotateBlink

    absl::Status EyeBlinkToRenderDataCalculator::Process(CalculatorContext* cc)
    {
        RenderData render_data;
        if (!cc->Inputs().Tag(kBlinkStreamTag).IsEmpty())
        {
            auto multi_face_blinks = cc->Inputs().Tag(kBlinkStreamTag).Get<std::vector<std::map<std::string, double> > >();
            if(!multi_face_blinks.empty())
            {
                auto blink = multi_face_blinks.at(0);
                auto threshold = blink.at("threshold");
                std::string left_blink     = blink.at("left") < threshold ? "Blink": "";
                std::string right_blink    = blink.at("right") < threshold ? "Blink": "";
            
                this->AnnotateBlink(render_data, left_blink, 0.08);
                this->AnnotateBlink(render_data, right_blink, 0.83);
            }
        }
        
        Packet packet = MakePacket<decltype(render_data)>(render_data).At(cc->InputTimestamp());
        cc->Outputs().Tag(kRenderDataStreamTag).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status EyeBlinkToRenderDataCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
