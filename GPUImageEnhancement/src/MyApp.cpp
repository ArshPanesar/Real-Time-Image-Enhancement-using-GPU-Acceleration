#include "pch.h"
#include "MyApp.h"

MyApp::MyApp()
{
}

MyApp::~MyApp()
{
}

void MyApp::Update()
{
    //SetImageBrightness(dev_img_ptr, app_img.GetSize(), 75);
    //DenoiseImageWithAverageFilter(dev_img_ptr, app_img.GetWidth(), app_img.GetHeight());
    //SetGrayLevelImageContrast(dev_img_ptr, app_img.GetWidth(), app_img.GetHeight());
    SetGrayscaledContrastWithJointHistogramEqualization(dev_img_ptr, app_img.GetWidth(), app_img.GetHeight());

    CopyToHost();

    app_img.SetRGB(host_img_ptr, app_img.GetWidth(), app_img.GetHeight());
    app_img.CopyToImage(&cimg);
}

void MyApp::UpdateWithMemTransfer()
{
    //SetImageBrightness(dev_img_ptr, app_img.GetSize(), 50);
    //SetGrayscaledContrastWithJointHistogramEqualization(dev_img_ptr, app_img.GetWidth(), app_img.GetHeight());
    //CPU_SetGrayscaledContrastWithJointHistogramEqualization(host_img_ptr, app_img.GetWidth(), app_img.GetHeight());
    //CopyToHost();

    //app_img.SetRGB(host_img_ptr, app_img.GetWidth(), app_img.GetHeight());
    //app_img.CopyToImage(&cimg);
}

void MyApp::RunNormContrastEnhance()
{
    SetGrayLevelImageContrast(dev_img_ptr, app_img.GetWidth(), app_img.GetHeight());
}

void MyApp::RunJointContrastEnhance()
{
    SetGrayscaledContrastWithJointHistogramEqualization(dev_img_ptr, app_img.GetWidth(), app_img.GetHeight());
}
