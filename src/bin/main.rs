use summa::errors::SummaResult;
use summa::Application;

fn main() -> SummaResult<()> {
    Application::proceed_args()
}
