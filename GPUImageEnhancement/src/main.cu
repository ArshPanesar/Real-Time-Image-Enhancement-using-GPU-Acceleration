#include "matrix_add.cuh"
#include "MyApp.h"
#include <CSVWriter.h>
#include "image_enhance.cuh"

using namespace cimg_library;
using namespace gpu_enhance;

void profile_workbench()
{
	CSVWriter csv(",");

	std::ifstream ifs;
	std::string line;
	ifs.open("input.txt");

	std::getline(ifs, line);
	std::string file = line;
	MyApp App;
	App.LoadImg(file);

	std::string img_res = std::to_string(App.GetAppImg().GetWidth()) + "x" + std::to_string(App.GetAppImg().GetHeight());

	double AvgDurationForNorm = 0.0;
	double AvgDurationForJoint = 0.0;
	int NumOfTests = 10;

	// Warmup
	App.UpdateWithMemTransfer();

	for (int i = 0; i < NumOfTests; ++i)
	{
		Timer t1;
		t1.Start("");
		App.RunNormContrastEnhance();
		t1.Stop();

		Timer t2;
		t2.Start("");
		App.RunJointContrastEnhance();
		t2.Stop();

		AvgDurationForNorm += (double)t1.GetDurationInMicroseconds();
		AvgDurationForJoint += (double)t2.GetDurationInMicroseconds();
	}
	AvgDurationForNorm /= (double)NumOfTests;
	AvgDurationForJoint /= (double)NumOfTests;

	csv.newRow() << img_res << AvgDurationForNorm << AvgDurationForJoint;
	std::cout << AvgDurationForNorm << " " << AvgDurationForJoint;
	csv.writeToFile("profiling/contrast_enhance_comparison.csv", true);
	
}

int main() {

	//profile_workbench();

	// CUDA Warmup
	runMatrixAdd(16);
	
	std::ifstream ifs;
	std::string line;
	ifs.open("input.txt");

	std::getline(ifs, line);
	std::string load_file = line;
	//std::getline(ifs, line);
	//std::string save_file = line;


	MyApp App;

	App.LoadImg(load_file);

	CImgDisplay main_disp(App.GetCImg(), "TestBed", 0);
	
	
	//App.UpdateWithMemTransfer();

	while (!main_disp.is_closed()) {
		main_disp.wait();

		if (main_disp.is_key1())
		{
			App.Update();
		}
		else if (main_disp.is_key2())
		{
			App.UpdateWithMemTransfer();
		}

		main_disp.display(App.GetCImg());
	}

	//App.SaveImg(save_file);

	return 0;
}
