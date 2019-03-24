#include <iostream>
#include <opencv2/opencv.hpp>


#include "Common/Timer.h"
#include "Database/Database.h"
#include "Feature/FeatureUtils.h"
#include "Feature/FeatureExtraction.h"

using namespace MonocularSfM;



/**
 * @brief loadImages        : 提取该文件夹下的所有图片
 *
 *
 * @param images_path       : 图片所在的文件夹
 */
std::vector<cv::String> LoadImages(const std::string& images_path);

/**
 * @brief isImagePath       : 判断给定的路径是不是图片
 *
 *
 * @param image_path        : 图片路径
 */
bool IsImagePath(const std::string& image_path);
cv::String UnionImagePath(const std::string& images_path, const cv::String& image_path);
std::string GetImageName(const std::string& full_path);

/**
 * @brief scaleImage            :   将图片的最大尺寸缩放到max_image_size
 *
 *
 * @param src                   :   源图片
 * @param dst                   :   缩放之后的图片
 * @param max_image_size        :   图片的最大尺寸
 * @param scale_x               :   在x轴方向上的缩放比例
 * @param scale_y               :   在y轴方向上的缩放比例
 */
void ScaleImage(cv::Mat& src, cv::Mat& dst, const int& max_image_size, double& scale_x, double& scale_y);



void L1RootNormalized(cv::Mat& descriptors);
void L2Normalized(cv::Mat& descriptors);
void RootSIFTNormalized(cv::Mat& descriptors);


void FeatureExtractorCPU::RunExtraction()
{

    Database database;
    database.Open(database_path_);

    std::vector<cv::String> images = LoadImages(images_path_);

    for(size_t i = 0; i < images.size(); ++i)
    {

        Timer timer;
        timer.Start();

        const cv::String& image_path = images[i];
        std::cout << "ExtractFeature : " << i <<  " ... "  << std::endl;
        std::cout << "\t " << image_path << std::endl;;

        database.BeginTransaction();
        Database::Image db_image;

        if(!database.ExistImageByName(image_path))
        {
            db_image.id = i;
            db_image.name = GetImageName(image_path);
            database.WriteImage(db_image, true);
        }
        else
        {
            db_image = database.ReadImageByName(image_path);
        }

        const bool exist_keypoints = database.ExistKeyPoints(db_image.id);
        const bool exist_descriptors = database.ExistDescriptors(db_image.id);
        const int exist_keypoints_and_descriptor = (static_cast<int>(exist_keypoints) << 1) + static_cast<int>(exist_descriptors);

        /// (1 << 1) + 1 = 3
        /// (1 << 1) + 0 = 2
        /// (0 << 1) + 1 = 1
        /// (0 << 0) + 0 = 0
        assert(exist_keypoints_and_descriptor != 1);
        assert(exist_keypoints_and_descriptor != 2);
        assert(exist_keypoints_and_descriptor == 0 || exist_keypoints_and_descriptor == 3);

        if(exist_keypoints_and_descriptor == 3)
        {
            std::cout << "\t Alread exist,  continue." << std::endl;
            database.EndTransaction();
            continue;
        }


        cv::Mat image;
        cv::Mat scaled_image;
        cv::Mat gray_image;
        image = cv::imread(image_path);

        cv::cvtColor(image, gray_image, cv::COLOR_BGRA2GRAY);
        double scale_x;
        double scale_y;
        ScaleImage(gray_image, scaled_image, max_image_size_, scale_x, scale_y);

        int width = scaled_image.cols;
        int height = scaled_image.rows;


        std::vector<cv::KeyPoint> scaled_kpts;
        std::vector<cv::KeyPoint> kpts;
        std::vector<cv::Vec3b> kpts_color;
        cv::Mat descriptors;

        FeatureUtils::ExtractFeature(scaled_image, scaled_kpts, descriptors, max_num_features_);

        std::cout << "\t num : " << scaled_kpts.size() << std::endl;;
        std::cout << "\t ";
        timer.PrintSeconds();
        std::cout << std::endl;

        kpts.resize(scaled_kpts.size());
        kpts_color.resize(scaled_kpts.size());
        const double inv_scale_x = 1.0 / scale_x;
        const double inv_scale_y = 1.0 / scale_y;
        const double inv_scale_xy = (inv_scale_x + inv_scale_y) / 2.0f;
        for(size_t i = 0; i < scaled_kpts.size(); ++i)
        {
            kpts[i].pt.x = scaled_kpts[i].pt.x * inv_scale_x;
            kpts[i].pt.y = scaled_kpts[i].pt.y * inv_scale_y;
            kpts[i].size = scaled_kpts[i].size * inv_scale_xy;
            kpts[i].angle = scaled_kpts[i].angle;

            kpts_color[i] = image.at<cv::Vec3b>(kpts[i].pt.y, kpts[i].pt.x);
        }

        if(normalization_ == Normalization::L1_ROOT)
        {
            L1RootNormalized(descriptors);
        }
        else  if(normalization_ == Normalization::L2)
        {
            L2Normalized(descriptors);
        }
        else
        {
            assert(false);
        }

        database.WriteKeyPoints(db_image.id, kpts);
        database.WriteKeyPointsColor(db_image.id, kpts_color);
        database.WriteDescriptors(db_image.id, descriptors);

        database.EndTransaction();
    }

}





std::vector<cv::String> LoadImages(const std::string& images_path)
{
    std::vector<cv::String> tmp_images;
    std::vector<cv::String> images;
    cv::glob(images_path, tmp_images);
    for(size_t i = 0; i < tmp_images.size(); ++i)
    {
        const cv::String& image_path = tmp_images[i];
        if(not IsImagePath(image_path))
            continue;

        images.push_back(image_path);
    }
    return images;
}

bool IsImagePath(const std::string& image_path)
{
    // 常见的图片后缀名
    const std::string suffixs[] = {"jpg", "jpe", "jpeg", "png", "pbm", "pgm", "ppm", "bmp", "dib", "tif", "tiff", "jp2"};

    bool is_image_path = false;
    for(const std::string& suffix : suffixs)
    {
        const int len1 = image_path.size();
        const int len2 = suffix.size();
        if(len1 < len2)
            continue;
        bool is_current_suffix = true;
        for(int i = 0; i < len2; ++i)
        {
            // 从后往前比较
            if(tolower(image_path[len1 - i - 1]) != suffix[len2 - i - 1])
            {
                is_current_suffix = false;
                break;
            }
        }
        // 只要符合一个后缀，那么就是图片
        if(is_current_suffix)
        {
            is_image_path = true;
            break;
        }
    }
    return is_image_path;

}
cv::String UnionImagePath(const std::string& images_path, const cv::String& image_path)
{
    if(images_path[images_path.size() - 1] == '/')
    {
        return cv::String(images_path) + image_path;
    }
    else
    {
        return cv::String(images_path + "/") + image_path;
    }
}

std::string GetImageName(const std::string& full_path)
{
    int i = full_path.size() - 1;
    while(i >= 0 && full_path[i] != '/')
        i--;
    assert(i >= 0);
    return full_path.substr(i + 1, full_path.size());
}
void ScaleImage(cv::Mat& src, cv::Mat& dst, const int& max_image_size, double& scale_x, double& scale_y)
{
    if(max_image_size < src.rows || max_image_size < src.cols)
    {
        const int width = src.cols;
        const int height = src.rows;
        const double scale = max_image_size * 1.0 / std::max(width, height);
        const int new_width = width * scale;
        const int new_height = height * scale;

        scale_x = new_width * 1.0 / width;
        scale_y = new_height * 1.0 / height;

        cv::resize(src, dst, cv::Size(new_width, new_height));
    }
    else
    {
        scale_x = 1.0;
        scale_y = 1.0;
        dst = src.clone();
    }
}

void L1RootNormalized(cv::Mat& descriptors)
{
    for(size_t i = 0; i < descriptors.rows; ++i)
    {
        cv::Mat row = descriptors.rowRange(i, i + 1);
        const double norm_l1 = cv::norm(row, cv::NORM_L1);
        row /= norm_l1;
        cv::sqrt(row, row);

    }
}
void L2Normalized(cv::Mat& descriptors)
{
    for(size_t i = 0; i < descriptors.rows; ++i)
    {
        cv::Mat row = descriptors.rowRange(i, i + 1);
        const double norm_l2 = cv::norm(row, cv::NORM_L2);
        row /= norm_l2;
        /* cv::normalize(row, row, 1, 0, cv::NORM_L2); */

    }
}

void RootSIFTNormalized(cv::Mat& descriptors)
{
    assert(descriptors.type() == CV_32F);
    for(size_t i = 0; i < descriptors.rows; ++i)
    {
        cv::Mat desc = descriptors.rowRange(i, i + 1);
        double sum = 0;
        for(size_t j = 0; j < desc.cols; ++j)
        {
            sum += desc.at<float>(i, j);
        }
        for(size_t j = 0; j < desc.cols; ++j)
        {
            desc.at<float>(i, j) /= sum;
            desc.at<float>(i, j) = sqrt(desc.at<float>(i, j));
        }
    }
}
