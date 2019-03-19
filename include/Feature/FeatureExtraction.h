#ifndef __FEATURE_EXTRACTION_H__
#define __FEATURE_EXTRACTION_H__

#include <string>


namespace MonocularSfM
{
class FeatureExtractor
{
public:

    ///特征描述子的归一化方式
    enum class Normalization
    {
        L1_ROOT,
        L2,
        ROOT_SIFT
    };
public:

    /**
     * @param database_path         :	数据库路径
     * @param images_path           :   图片所在的文件夹
     * @param max_image_size        :   最大的图片尺寸，如果超过这个尺寸，那么会将图片的最大尺寸降低到max_image_size, 然后在降采样的图片上进行特征提取
     * @param max_num_features      :   最多提取的特征点的个数
     * @param normalization         :   特征描述子的归一化方式
     */
    FeatureExtractor(const std::string& database_path,
                     const std::string& images_path,
                     const int& max_image_size = 3200,
                     const int& max_num_features = 10240,
                     const Normalization& normalization = Normalization::L1_ROOT)
                    : database_path_(database_path),
                      images_path_(images_path),
                      max_image_size_(max_image_size),
                      max_num_features_(max_num_features),
                      normalization_(normalization) {}

    virtual void RunExtraction() = 0;

protected:
    std::string database_path_;
    std::string images_path_;
    int max_image_size_;
    int max_num_features_;
    Normalization normalization_;
};



class FeatureExtractorCPU : public FeatureExtractor
{
public:
    using FeatureExtractor::FeatureExtractor;
    void RunExtraction();
};



// TODO :使用SiftGPU在GPU上提取特征
class FeatureExtractorGPU : public FeatureExtractor
{
public:
    using FeatureExtractor::FeatureExtractor;
    virtual void RunExtraction() = 0;
};





} // namespace MonocularSfM

#endif // __FEATURE_EXTRACTION_H__

