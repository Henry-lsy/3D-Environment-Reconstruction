#include "feature_extract.hpp"
#include "feature_match.hpp"
#include "pose_estimation.hpp"
#include "trianglation.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/viz.hpp>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;

std::vector<cv::Mat> readImagesFromFolder(const std::string& folderPath) {
    int file_num = 0;
    // Iterate through the directory and count the .png files
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            file_num++;  // Count the .png files
        }
    }
    std::vector<cv::Mat> images;
    for (int i = 0; i < file_num; i++)
    {
        auto imagePath = folderPath + "/" + std::to_string(i) + ".png";
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (!image.empty()) {
            images.push_back(image);
        }
    }
    return images;
}

void visualizeCameraPosesAndPoints(const vector<cv::Mat>& rotations, const vector<cv::Mat>& translations, const std::vector<cv::Point3d> points) {
    // Create a Viz window
    viz::Viz3d window("Camera Pose and 3D Points Visualization");

    // Set up world coordinate system
    viz::WCoordinateSystem worldCoordinateSystem(1.0);  // Size of the axes: 1.0 unit
    window.showWidget("World Coordinate System", worldCoordinateSystem);

    // Initialize the transformation for the first camera pose (identity matrix)
    Affine3d currentPose = Affine3d();  // Default constructor initializes to identity matrix

    // Loop through the rotation and translation matrices
    for (size_t i = 0; i < rotations.size(); ++i) {
        // Get the current rotation matrix and translation vector
        const cv::Mat& R = rotations[i];
        const cv::Mat& t = translations[i];

        // Create the affine transformation for the current camera pose
        Affine3d cameraPose(R, t);

        // Apply the current camera pose relative to the previous camera pose
        currentPose = currentPose * cameraPose;  // Accumulate the transformation

        // Display the camera's coordinate system at the given pose
        string widgetName = "Camera Pose " + to_string(i);  // Name each camera pose widget uniquely
        window.showWidget(widgetName, viz::WCoordinateSystem(), currentPose);
    }

    // Visualize the 3D points using OpenCV's Viz module
    std::vector<cv::Point3f> scenePoints;
    for (const auto& point : points) {
        scenePoints.push_back(cv::Point3f(point.x, point.y, point.z));
    }
    viz::WCloud cloud(scenePoints);  // Create a cloud from the points
    window.showWidget("3D Points", cloud);

    // Start the visualization window loop
    window.spin();
}

// void init_structure(
// 	Mat K,
// 	vector<vector<KeyPoint>>& key_points_for_all,
// 	vector<vector<Vec3b>>& colors_for_all,
// 	vector<vector<DMatch>>& matches_for_all,
// 	vector<Point3f>& structure,
// 	vector<vector<int>>& correspond_struct_idx,
// 	vector<Vec3b>& colors,
// 	vector<Mat>& rotations,
// 	vector<Mat>& motions
// )
// {
// 	// 计算头两幅图像之间的变换矩阵
// 	vector<Point2f> p1, p2;
// 	vector<Vec3b> c2;
// 	Mat R, T;	// 旋转矩阵和平移向量
// 	Mat mask;	// mask中大于零的点代表匹配点，等于零的点代表失配点
// 	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
// 	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
// 	find_transform(K, p1, p2, R, T, mask);	// 三角分解得到R， T 矩阵

// 	// 对头两幅图像进行三维重建
// 	maskout_points(p1, mask);
// 	maskout_points(p2, mask);
// 	maskout_colors(colors, mask);

// 	Mat R0 = Mat::eye(3, 3, CV_64FC1);
// 	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
// 	reconstruct(K, R0, T0, R, T, p1, p2, structure);	// 三角化
// 	// 保存变换矩阵
// 	rotations = { R0, R };
// 	motions = { T0, T };

// 	// 将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
// 	correspond_struct_idx.clear();
// 	correspond_struct_idx.resize(key_points_for_all.size());
// 	for (int i = 0; i < key_points_for_all.size(); ++i)
// 	{
// 		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
// 	}

// 	// 填写头两幅图像的结构索引
// 	int idx = 0;
// 	vector<DMatch>& matches = matches_for_all[0];
// 	for (int i = 0; i < matches.size(); ++i)
// 	{
// 		if (mask.at<uchar>(i) == 0)
// 		{
// 			continue;
// 		}

// 		correspond_struct_idx[0][matches[i].queryIdx] = idx;	// 如果两个点对应的idx 相等 表明它们是同一特征点 idx 就是structure中对应的空间点坐标索引
// 		correspond_struct_idx[1][matches[i].trainIdx] = idx;
// 		++idx;
// 	}
// }
inline cv::Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}


// void init_structure(
// 	Mat K,
// 	vector<vector<KeyPoint>>& key_points_for_all,
// 	vector<vector<Vec3b>>& colors_for_all,
// 	vector<vector<DMatch>>& matches_for_all,
// 	vector<Point3f>& structure,
// 	vector<vector<int>>& correspond_struct_idx,
// 	vector<Vec3b>& colors,
// 	vector<Mat>& rotations,
// 	vector<Mat>& motions
// )
// {
// 	// 计算头两幅图像之间的变换矩阵
// 	vector<Point2f> p1, p2;
// 	vector<Vec3b> c2;
// 	Mat R, T;	// 旋转矩阵和平移向量
// 	Mat mask;	// mask中大于零的点代表匹配点，等于零的点代表失配点
// 	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
// 	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);
// 	find_transform(K, p1, p2, R, T, mask);	// 三角分解得到R， T 矩阵
// 	std::cout << "R is " << R << std::endl; 
// 	std::cout << "T is " << T << std::endl; 
// 	// 对头两幅图像进行三维重建
// 	maskout_points(p1, mask);
// 	maskout_points(p2, mask);
// 	maskout_colors(colors, mask);

// 	Mat R0 = Mat::eye(3, 3, CV_64FC1);
// 	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
// 	reconstruct(K, R0, T0, R, T, p1, p2, structure);	// 三角化
// 	// 保存变换矩阵
// 	rotations = { R0, R };
// 	motions = { T0, T };

// 	// 将correspond_struct_idx的大小初始化为与key_points_for_all完全一致
// 	correspond_struct_idx.clear();
// 	correspond_struct_idx.resize(key_points_for_all.size());
// 	for (int i = 0; i < key_points_for_all.size(); ++i)
// 	{
// 		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
// 	}

// 	// 填写头两幅图像的结构索引
// 	int idx = 0;
// 	vector<DMatch>& matches = matches_for_all[0];
// 	for (int i = 0; i < matches.size(); ++i)
// 	{
// 		if (mask.at<uchar>(i) == 0)
// 		{
// 			continue;
// 		}

// 		correspond_struct_idx[0][matches[i].queryIdx] = idx;	// 如果两个点对应的idx 相等 表明它们是同一特征点 idx 就是structure中对应的空间点坐标索引
// 		correspond_struct_idx[1][matches[i].trainIdx] = idx;
// 		++idx;
// 	}
// }

void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3f>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points)
{
	object_points.clear();
	image_points.clear();

	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx < 0)	// 表明跟前一副图像没有匹配点
		{
			continue;
		}

		object_points.push_back(structure[struct_idx]);
		image_points.push_back(key_points[train_idx].pt);	// train中对应关键点的坐标 二维
	}
}

int main() {
    std::string folderPath = "../Coding_Assignment_Data/Foutain_Comp"; // 替换为您的图片文件夹路径
    std::vector<cv::Mat> images = readImagesFromFolder(folderPath);

    if (images.size() < 2) {
        std::cerr << "At least two images are required." << std::endl;
        return 1;
    }
    // auto features = extractFeaturesORB(images, true);

    std::vector<std::vector<cv::KeyPoint>> key_points_for_all;
    std::vector<cv::Mat> descriptor_for_all;
    std::vector<std::vector<cv::Vec3b>> colors_for_all;
    std::vector<std::vector<DMatch>> matches_for_all;
    // Step 1: extracting features and matching.
    extract_features(images, key_points_for_all, descriptor_for_all, colors_for_all);
    match_features(descriptor_for_all, matches_for_all);
    visualizeGoodMatches(images, key_points_for_all, descriptor_for_all, colors_for_all, matches_for_all);

    // Step 2: Compute point clouds by the first two images. 
    Mat R, T;	// 旋转矩阵和平移向量
    pose_estimation_2d2d(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], R, T);
    // Step 2.1: Trianglation

    vector<Point3d> points;
    triangulation(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], R, T, points);

    vector<cv::Mat> Rs, Ts;
    Rs.push_back(R);
    Ts.push_back(T);

    visualizeCameraPosesAndPoints(Rs, Ts, points);



    return 0;
}