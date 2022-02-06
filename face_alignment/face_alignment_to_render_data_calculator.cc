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
        constexpr char kAlignmentStreamTag[]  = "ALIGNMENT";
        constexpr char kRenderDataStreamTag[] = "RENDER";
    } // namespace

    /**
     * @brief Annotate Detected Face Alignment
     * 
     * INPUTS:
     *      ALIGNMENT - Alignments (std::vector<std::map<std::string, double> >)
     * OUTPUTS:
     *      RENDER - Render Data to be render by OverlayRenderer (RenderData)
     * 
     * Example:
     * 
     * node {
     *   calculator: "FaceAlignmentToRenderDataCalculator"
     *   input_stream: "ALIGNMENT:multi_face_alignments"
     *   output_stream: "RENDER:alignment_render_data"
     * }
     * 
     */
    class FaceAlignmentToRenderDataCalculator: public CalculatorBase
    {
    private:
        void AnnotateAlignment(RenderData& render_data, std::string alignment, int left_pos);
    public:
        FaceAlignmentToRenderDataCalculator() = default;
        ~FaceAlignmentToRenderDataCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;

    };
    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(FaceAlignmentToRenderDataCalculator);

    absl::Status FaceAlignmentToRenderDataCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag(kAlignmentStreamTag).Set<std::vector<std::map<std::string, double> > >();
        cc->Outputs().Tag(kRenderDataStreamTag).Set<RenderData>();
        return absl::OkStatus();
    }

    absl::Status FaceAlignmentToRenderDataCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }

    void FaceAlignmentToRenderDataCalculator::AnnotateAlignment(RenderData& render_data, std::string alignment, int left_pos)
    {
        auto annotation = render_data.add_render_annotations();
        if(alignment == "Neutral")
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
        text->set_font_height(40);
        text->set_font_face(0);
        text->set_display_text(alignment);
        text->set_normalized(false);
        text->set_left(left_pos);
        // Normalized coordinates must be between 0.0 and 1.0, if they are used.
        text->set_baseline(300);
    } // AnnotateAlignment()

    absl::Status FaceAlignmentToRenderDataCalculator::Process(CalculatorContext* cc)
    {
        RenderData render_data;
        if (!cc->Inputs().Tag(kAlignmentStreamTag).IsEmpty())
        {
            auto multi_face_alignments = cc->Inputs().Tag(kAlignmentStreamTag).Get<std::vector<std::map<std::string, double> > >();
            if(!multi_face_alignments.empty())
            {
                auto alignment = multi_face_alignments.at(0);
                std::string hor_align =    alignment.at("horizontal_align") >= 0.3 ? "Right":
                                            alignment.at("horizontal_align") <= -0.3 ? "Left":
                                            "Neutral";
                std::string ver_align =    alignment.at("vertical_align") >= 0.6 ? "Down":
                                            alignment.at("vertical_align") <= -0.05 ? "Up":
                                            "Neutral";
                
                this->AnnotateAlignment(render_data, hor_align, 50);
                this->AnnotateAlignment(render_data, ver_align, 450);
            }
        }
        
        Packet packet = MakePacket<decltype(render_data)>(render_data).At(cc->InputTimestamp());
        cc->Outputs().Tag(kRenderDataStreamTag).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status FaceAlignmentToRenderDataCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
