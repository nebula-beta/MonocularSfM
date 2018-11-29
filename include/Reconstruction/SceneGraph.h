#ifndef __SCENE_GRAPH_H__
#define __SCENE_GRAPH_H__


#include "Common/Types.h"
#include "Database/Database.h"


namespace MonocularSfM
{


/**
 *  SceneGraph用来快速
 *  (1) 找到每张图片有多个个观察点（num_observations)
 *  (2) 找到每张图片与其它所有图片总共有多少个对应点(num_correspondences)
 *  (3) 找到两幅图像之间的匹配点
 *  (4) 某图像某个特征点的所有匹配点（分布在多幅图像上）
 */


class SceneGraph
{
public:
    struct Correspondence
    {
        Correspondence() : image_id(INVALID), point2D_idx(INVALID) { }
        Correspondence(const image_t image_id, point2D_t point2D_idx)
            : image_id(image_id), point2D_idx(point2D_idx){ }
        image_t image_id;
        point2D_t point2D_idx;
    };


    SceneGraph() { }

    void Load(const cv::Ptr<Database> database, const size_t min_num_matches);
    void Finalize();



    size_t NumImages() const;

    /**
     * 判断SceneGraph中是否存在image_id这个图像
     * @param image_id      : 图片id
     * @return              : 存在， 返回true； 否则，返回false.
     */
    bool ExistsImage(const image_t image_id) const;


    /**
     * 找到每张图片有多个个观察点（num_observations)
     * @param image_id      : 图片id
     * @return              : num_observations
     */
    point2D_t NumObservationsForImage(image_t image_id) const;

    /**
     * 找到每张图片与其它所有图片总共有多少个对应点(num_correspondences)
     * @param image_id      : 图片id
     * @return              : num_correspondences
     */
    point2D_t NumCorrespondencesForImage(image_t image_id) const;

    /**
     * 找到两幅图像之间总共有多少个对应点
     * @param image_id1     : 图像id1
     * @param image_id2     : 图像id2
     * @return              : image_id1和image_id2之间的特征点匹配数
     */
    point2D_t NumCorrespondencesBetweenImages(const image_t image_id1, const image_t image_id2) const;




    /**
     * 向SceneGraph中添加图像
     * @param image_id          : 图像id
     * @param num_points2D      : 该图像特征点的数量
     */
    void AddImage(const image_t image_id, const size_t num_points2D);

    /**
     * 向SceneGraph添加两幅图像的特征点匹配
     * 需要注意的是，
     *  (1) 已经调用过AddImage函数， 向SceneGraph中添加过image_id1和image_id2
     *  (2) matches是经过交叉验证的(计算匹配时，默认已验证）
     * @param image_id1     : 第一幅图像的id
     * @param image_id2     : 第二幅图像的id
     * @param matches       : 经过交叉验证的特征匹配
     */
    void AddCorrespondences(const image_t image_id1,
                            const image_t image_id2,
                            const std::vector<cv::DMatch>& matches);

    /**
     * 找到第image_id1张图片的第point2D_idx个特征点的所有匹配点（分布在多幅图像上）
     * @param image_id      : 图像id
     * @param point2D_idx   : 特征点序号
     * @return              : 返回该特征点在其它图像上的所有匹配点
     */
    const std::vector<typename SceneGraph::Correspondence> FindCorrespondences(
            const image_t image_id, const point2D_t point2D_idx) const;

    /**
     * 获得两幅图像之间的匹配
     * @param image_id1     : 第一幅图像的id
     * @param image_id2     : 第二幅图像的id
     * @return              : 返回两幅图像之间的匹配
     */
    std::vector<cv::DMatch> FindCorrespondencesBetweenImages(
            const image_t image_id1, const image_t image_id2) const;

    /**
     * 判断第image_id1张图片的第point2D_idx个特征点是否存在匹配点
     * @param image_id      : 图像id
     * @param point2D_idx   : 特征点序号
     * @return              : 存在匹配点， 返回true; 否则， 返回false.
     */
    bool HasCorrespondences(const image_t image_id, const point2D_t point2D_idx) const;

    /**
     * 判断第image_id1张图片的第point2D_idx个特征点是否是TwoViewObservation
     * TwoViewObservation的含义是:
     *    第image_id1张图片的第point2D_idx个特征点只与一张图片的特征点发生匹配（设为corr_image_id和corr_point2D_idx)
     *    且第corr_image_id张图片的第corr_point2D_idx特征点也只与第image_id1张图片的第point2D_idx个特征点发生匹配
     * TwoViewObservation生成的Track只有两个元素
     * 不能为后续的图片提供2D-3D对应
     *
     * @param image_id      : 图像id
     * @param point2D_idx   : 特征点序号
     * @return              : 如果是TwoViewObservation， 返回true; 否则， 返回false.
     */
    bool IsTwoViewObservation(const image_t image_id, const point2D_t point2D_idx) const;




    std::vector<image_t> GetAllImageIds() const;

    const std::unordered_map<image_pair_t, point2D_t> ImagePairs();

private:
    struct Image
    {
        // num_observations表示在该图像的特征点中,有多少个点能够找到匹配点
        point2D_t num_observations = 0;
        // 将与其它图像的匹配点累计起来, 总共多少个匹配点
        point2D_t num_correspondences = 0;

        ///corrs[i] 表示第i个特征点的与其它图片的哪些特征点是匹配点
        std::vector<std::vector<Correspondence>> corrs;
    };


    // 场景图的结点
    std::unordered_map<image_t, typename SceneGraph::Image> images_;


    // 图像对之间有效的的匹配点的数量
    std::unordered_map<image_pair_t, point2D_t> image_pairs_;


};





} // namespace MonocularSfM

#endif // __SCENE_GRAPH_H__
