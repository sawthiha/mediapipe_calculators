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
        constexpr char korientationStreamTag[]  = "orientation";
        constexpr char kRenderDataStreamTag[] = "RENDER";
    } // namespace

    /**
     * @brief Annotate Detected Face orientation
     * 
     * INPUTS:
     *      orientation - orientations (std::vector<std::map<std::string, double> >)
     * OUTPUTS:
     *      RENDER - Render Data to be render by OverlayRenderer (RenderData)
     * 
     * Example:
     * 
     * node {
     *   calculator: "FaceOrientationToRenderDataCalculator"
     *   input_stream: "orientation:multi_face_orientations"
     *   output_stream: "RENDER:orientation_render_data"
     * }
     * 
     */
    class FaceOrientationToRenderDataCalculator: public CalculatorBase
    {
    private:
        void Annotateorientation(RenderData& render_data, std::string orientation, double left_pos);
    public:
        FaceOrientationToRenderDataCalculator() = default;
        ~FaceOrientationToRenderDataCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;

    };
    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(FaceOrientationToRenderDataCalculator);

    absl::Status FaceOrientationToRenderDataCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag(korientationStreamTag).Set<std::vector<std::map<std::string, double> > >();
        cc->Outputs().Tag(kRenderDataStreamTag).Set<RenderData>();
        return absl::OkStatus();
    }

    absl::Status FaceOrientationToRenderDataCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }

    void FaceOrientationToRenderDataCalculator::Annotateorientation(RenderData& render_data, std::string orientation, double left_pos)
    {
        auto annotation = render_data.add_render_annotations();
        if(orientation == "Neutral")
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
        annotation->set_thickness(4);
        auto text = annotation->mutable_text();
        text->set_font_height(0.04);
        text->set_font_face(0);
        text->set_display_text(orientation);
        text->set_normalized(true);
        text->set_left(left_pos);
        // Normalized coordinates must be between 0.0 and 1.0, if they are used.
        text->set_baseline(0.2);
    } // Annotateorientation()

    absl::Status FaceOrientationToRenderDataCalculator::Process(CalculatorContext* cc)
    {
        RenderData render_data;
        if (!cc->Inputs().Tag(korientationStreamTag).IsEmpty())
        {
            auto multi_face_orientations = cc->Inputs().Tag(korientationStreamTag).Get<std::vector<std::map<std::string, double> > >();
            if(!multi_face_orientations.empty())
            {
                auto orientation = multi_face_orientations.at(0);
                std::string hor_align =    orientation.at("horizontal_align") >= 0.3 ? "Right":
                                            orientation.at("horizontal_align") <= -0.3 ? "Left":
                                            "Neutral";
                std::string ver_align =    orientation.at("vertical_align") >= 0.6 ? "Down":
                                            orientation.at("vertical_align") <= -0.05 ? "Up":
                                            "Neutral";
                
                this->Annotateorientation(render_data, hor_align, 0.05);
                this->Annotateorientation(render_data, ver_align, 0.6);
            }
        }
        
        Packet packet = MakePacket<decltype(render_data)>(render_data).At(cc->InputTimestamp());
        cc->Outputs().Tag(kRenderDataStreamTag).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status FaceOrientationToRenderDataCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
