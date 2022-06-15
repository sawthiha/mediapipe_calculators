#include <array>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/calculators/custom/util/constant_matrix_calculator.pb.h"

namespace mediapipe
{
    /**
     * @brief Constant Matrix Stream Calculator
     * 
     * OUTPUTS:
     *      TICK - For synchronization purpose
     * 
     * Example:
     * 
     * # Constant Matrix Calculator
     *  node  {
     *      calculator: "ConstantMatrixCalculator"
     *      input_stream: "TICK:sync_stream"
     *      output_stream: "MATRIX:matrix"
     *      node_options: {
     *          [type.googleapis.com/mediapipe.ConstantMatrixCalculatorOptions] {
     *              # In row-major format
     *              1.0, 0.0, 0.0, 0.0,
     *              0.0, 1.0, 0.0, 0.0,
     *              0.0, 0.0, 1.0, 0.0,
     *              0.0, 0.0, 0.0, 1.0
     *          }
     *      }
     *  }
     * 
     */
    class ConstantMatrixCalculator: public CalculatorBase
    {
    private:
        ConstantMatrixCalculatorOptions m_options;

    public:
        ConstantMatrixCalculator() = default;
        ~ConstantMatrixCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(ConstantMatrixCalculator);

    absl::Status ConstantMatrixCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Tag("TICK").SetAny();
        cc->Outputs().Tag("MATRIX").Set<std::array<float, 16>>();
        return absl::OkStatus();
    }

    absl::Status ConstantMatrixCalculator::Open(CalculatorContext* cc)
    {
        m_options = cc->Options<ConstantMatrixCalculatorOptions>();
        assert(m_options.values_size() == 16);
        return absl::OkStatus();
    }

    absl::Status ConstantMatrixCalculator::Process(CalculatorContext* cc)
    {
        auto values = m_options.values().data();
        auto matrix = std::make_unique<std::array<float, 16>>();
        std::copy(values, values + 16, matrix->data());

        cc->Outputs().Tag("MATRIX").Add(matrix.release(), cc->InputTimestamp());

        return absl::OkStatus();
    } // Process()

    absl::Status ConstantMatrixCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe

