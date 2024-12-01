#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <utility>

void extract_features(
    std::vector<cv::Mat> images,
    std::vector<std::vector<cv::KeyPoint>>& key_points_for_all,
    std::vector<cv::Mat>& descriptor_for_all,
    std::vector<std::vector<cv::Vec3b>>& colors_for_all
) {
    key_points_for_all.clear();
    descriptor_for_all.clear();

    // 创建 SIFT 特征提取器
    Ptr<cv::Feature2D> sift = cv::SIFT::create(0, 3, 0.04, 10);
	  // Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(0, 3, 0.04, 10);

    // 遍历图像列表，提取每张图像的特征点和描述符
    for (size_t i = 0; i < images.size(); ++i) {
        const cv::Mat& image = images[i];
        if (image.empty()) {
            std::cerr << "Warning: Skipping empty image at index " << i << std::endl;
            continue;
        }
        std::cout << "Extracting features from image " << i << std::endl;

        // 检测和计算特征点和描述符
        std::vector<cv::KeyPoint> key_points;
        cv::Mat descriptor;
        sift->detectAndCompute(image, cv::noArray(), key_points, descriptor);

        // 如果特征点过少，则跳过该图像
        if (key_points.size() <= 10) {
            std::cerr << "Warning: Too few keypoints in image " << i << ", skipping..." << std::endl;
            continue;
        }

        // 保存特征点和描述符
        key_points_for_all.push_back(key_points);
        descriptor_for_all.push_back(descriptor);

        // 获取特征点的颜色信息
        std::vector<cv::Vec3b> colors(key_points.size());
        for (size_t j = 0; j < key_points.size(); ++j) {
            const cv::Point2f& p = key_points[j].pt;
            // 检查点是否在图像边界内
            if (p.x >= 0 && p.x < image.cols && p.y >= 0 && p.y < image.rows) {
                colors[j] = image.at<cv::Vec3b>(cv::Point(p.x, p.y));
            }
        }
        colors_for_all.push_back(colors);
    }
}

void match_features(const Mat& query, const Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	// 获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		// Rotio Test
		if (knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance)
		{
			continue;
		}

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist)
		{
			min_dist = dist;
		}
	}

	matches.clear();
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		// 排除不满足Ratio Test的点和匹配距离过大的点
		if (
			knn_matches[r][0].distance > 0.6 * knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 5 * max(min_dist, 10.0f)
			)
		{
			continue;
		}

		// 保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
}

void match_features(const vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	// n个图像，两两顺次有 n-1 对匹配
	// 1与2匹配，2与3匹配，3与4匹配，以此类推
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		match_features(descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
	}
}

void visualizeGoodMatches(const std::vector<cv::Mat>& images,
                          const std::vector<std::vector<cv::KeyPoint>>& key_points_for_all,
                          const std::vector<cv::Mat>& descriptor_for_all,
                          const std::vector<std::vector<cv::Vec3b>>& colors_for_all,
                          const std::vector<std::vector<cv::DMatch>>& matches_for_all) {
    for (size_t i = 0; i < images.size(); ++i) {
        if (i + 1 < images.size()) { // 确保有下一张图片与之匹配
            const cv::Mat& img_1 = images[i];
            const cv::Mat& img_2 = images[i + 1];
            const std::vector<cv::KeyPoint>& keypoints_1 = key_points_for_all[i];
            const std::vector<cv::KeyPoint>& keypoints_2 = key_points_for_all[i + 1];
            const cv::Mat& descriptors_1 = descriptor_for_all[i];
            const cv::Mat& descriptors_2 = descriptor_for_all[i + 1];
            const std::vector<cv::Vec3b>& colors_1 = colors_for_all[i];
            const std::vector<cv::Vec3b>& colors_2 = colors_for_all[i + 1];
            const std::vector<cv::DMatch>& matches = matches_for_all[i];

            cv::Mat img_match;
            cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            // 显示匹配结果
            cv::imshow("Good Matches", img_match);
            cv::waitKey(0);
        }
    }
}