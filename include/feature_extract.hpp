#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp> // 需要 xfeatures2d 模块

using namespace std;
using namespace cv;
std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> extractFeaturesORB(const std::vector<cv::Mat>& images, bool visualization = false) {
    std::vector<std::pair<std::vector<cv::KeyPoint>, cv::Mat>> features;

    // 创建ORB特征检测器
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    for (const auto& image : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        // 检测关键点和计算描述符
        orb->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        // 将结果存储到features向量中
        features.emplace_back(keypoints, descriptors);

        // 可视化关键点及其描述符
        if (visualization) {
            cv::Mat outputImage;
            cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

            // 可视化描述符的示例：绘制描述符的方向
            for (const auto& keypoint : keypoints) {
                // 计算箭头的终点
                int length = 20; // 箭头长度
                int x_end = static_cast<int>(keypoint.pt.x + length * cos(keypoint.angle * CV_PI / 180.0));
                int y_end = static_cast<int>(keypoint.pt.y + length * sin(keypoint.angle * CV_PI / 180.0));

                // 绘制箭头
                cv::arrowedLine(outputImage, keypoint.pt, cv::Point(x_end, y_end), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }

            // 显示结果
            cv::imshow("Keypoints and Descriptors", outputImage);
            cv::waitKey(0);  // 等待按键
        }
    }

    return features;
}