#include "Reconstruction/Mapper.h"

using namespace MonocularSfM;



Mapper::Mapper(const std::string& database_path, const Mapper::Config &config)
{
    database_path_ = database_path;
    database_ = cv::Ptr<Database>(new Database());
    database_->Open(database_path_);

    config_ = config;



}
Mapper::~Mapper()
{
    database_->Close();
}

void Mapper::SetUp()
{
    K_ = (cv::Mat_<float>(3, 3) << config_.fx, 0, config_.cx,
                                   0, config_.fy, config_.cy,
                                   0, 0, 1);
    dist_coef_ = (cv::Mat_<double>(4, 1) << config_.k1, config_.k2,
                                            config_.p1, config_.p2);

    std::cout << "内参 K " << std::endl;
    std::cout << K_ << std::endl << std::endl;
    std::cout << "畸变参数 " << std::endl;
    std::cout << dist_coef_ << std::endl << std::endl;


	// 从数据库中加载图片， 构建场景图
    scene_graph_  = cv::Ptr<SceneGraph>(new SceneGraph());
    scene_graph_->Load(database_, config_.min_num_matches);
    std::vector<image_t> ids = scene_graph_->GetAllImageIds();

	// 初始化图片数据
    for(image_t id : ids)
    {
        Database::Image db_image = database_->ReadImageById(id);
        Image image(db_image.id, db_image.name);
        images_[id] = image;
    }
    num_registered_ = 0;
    num_point3D_index_ = 0;

    prev_num_register_images_ = 0;
    num_global_ba_ = 0;

}

void Mapper::LoadImageData(const image_t &image_id)
{
    Image& image = images_[image_id];

    if(image.Point2Ds().size() == 0)
    {
        cv::KeyPoint::convert(database_->ReadKeyPoints(image_id), image.Point2Ds());
        image.Colors() = database_->ReadKeyPointsColor(image_id);

        if(IsNeedUndistortFeature())
        {
            UndistortFeature(image_id);
        }
    }
}

void Mapper::DoMapper()
{
    Timer timer;
    timer.Start();


    SetUp();

    AsyncVisualization async_visualization;
    async_visualization.RunVisualizationThread();

    // 初始化
    int num_init_trials = 0;
    bool is_succeed = false;
    while(true)
    {
        std::vector<image_t> image_ids1 = FindFirstInitialImage();
        for(image_t image_id1 : image_ids1)
        {
            std::vector<image_t> image_ids2 = FindSecondInitialImage(image_id1);
            for(image_t image_id2 : image_ids2)
            {
                is_succeed = Initialize(image_id1, image_id2);
                ++num_init_trials;
                if(num_init_trials >= config_.max_initialize_num_trials || is_succeed)
                    break;

            }
            if(num_init_trials >= config_.max_initialize_num_trials || is_succeed)
                break;
        }
        if(num_init_trials >= config_.max_initialize_num_trials || is_succeed)
            break;
    }

    if(is_succeed)
    {
        async_visualization.ShowPointCloud(images_, map_points_, map_points_color_);
        async_visualization.ShowCameras(images_, registered_);


        GlobalBA();
        FilterAllMapPoints();
        prev_num_register_images_ = 1;
        num_global_ba_ = 1;
    }
    else
    {
        std::cout << "Initialize fail!!!" << std::endl;
    }




    // 注册下一张
    bool regis_next_success = is_succeed;
    int num_regis_trials = 0;
    while(regis_next_success)
    {
        std::cout << "Total registered image : " << num_registered_ << std::endl;

        std::vector<image_t> next_image_ids = FindNextImages();
        if(next_image_ids.size() == 0)
        {
            regis_next_success = false;
            break;
        }

        for(size_t regis_trial = 0; regis_trial < next_image_ids.size(); ++regis_trial)
        {
            image_t next_image_id = next_image_ids[regis_trial];
            regis_next_success = RegisterNextImage(next_image_id);
            num_regis_trials += 1;
            if(regis_next_success || num_regis_trials > config_.max_register_next_num_trials)
                break;
        }

        if(regis_next_success)
        {
            num_regis_trials = 0;
            async_visualization.ShowPointCloud(images_, map_points_, map_points_color_);
            async_visualization.ShowCameras(images_, registered_);


            if(num_registered_ >= config_.ba_images_ratio * prev_num_register_images_)
            {
                GlobalBA();
                FilterAllMapPoints();
                Retriangulate();
                prev_num_register_images_ = num_registered_;
                num_global_ba_ += 1;
            }
            MergeAllMapPoints();

            CompleteAllMapPoints();

        }
    }

    Summary();

    std::cout << "\t Total registered time: ";
    timer.PrintMinutes();

    async_visualization.WaitForVisualizationThread();

}

void Mapper::Summary()
{
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "--------------------- Summary --------------------- " << std::endl;
    std::cout << "\t total register image : " << num_registered_;
    std::cout << "\t total map points : " << map_points_.size() << std::endl;
    std::cout << "\t total global ba : " << num_global_ba_ << std::endl;
    std::cout << "\t ave map points error : " << GetAveMapPointsError() << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;

}


std::vector<image_t> Mapper::FindFirstInitialImage() const
{
    struct ImageInfo
    {
        image_t image_id;
        point2D_t num_correspondences;
    };

    std::vector<ImageInfo> image_infos;

    std::vector<image_t> image_ids = scene_graph_->GetAllImageIds();
    for(image_t image_id : image_ids)
    {
        // 初始化失败过的图片就不作为初始化
        if(num_trial_registrations_.count(image_id) > 0 &&
           num_trial_registrations_.at(image_id) > 0)
            continue;

        ImageInfo image_info;
        point2D_t num_correspondences = scene_graph_->NumCorrespondencesForImage(image_id);
        image_info.image_id = image_id;
        image_info.num_correspondences = num_correspondences;
        image_infos.push_back(image_info);
    }


    // 按照每张图片的correspondences, 从大到小排序
    std::sort(
          image_infos.begin(), image_infos.end(),
          [](const ImageInfo& image_info1, const ImageInfo& image_info2)
            {
                return image_info1.num_correspondences > image_info2.num_correspondences;
            }
           );

    image_ids.clear();
    for(const ImageInfo& image_info : image_infos)
    {
        image_ids.push_back(image_info.image_id);
    }

    return image_ids;
}


std::vector<image_t> Mapper::FindSecondInitialImage(image_t image_id) const
{
    std::unordered_map<image_t, point2D_t> num_correspondences;
    point2D_t num_points2D = database_->NumKeyPoints(image_id);
    for(point2D_t point2D_idx = 0; point2D_idx < num_points2D; ++point2D_idx)
    {
        const std::vector<SceneGraph::Correspondence>& corrs =
                scene_graph_->FindCorrespondences(image_id, point2D_idx);


        for(const SceneGraph::Correspondence& corr : corrs)
        {
            num_correspondences[corr.image_id] += 1;
        }
    }
    struct ImageInfo
    {
        image_t image_id;
        point2D_t num_correspondences;
    };

    std::vector<ImageInfo> image_infos;

    for(const auto& elem : num_correspondences)
    {
        if(num_trial_registrations_.count(elem.first) > 0 &&
           num_trial_registrations_.at(elem.first) > 0)
            continue;

        ImageInfo image_info;
        image_info.image_id = elem.first;
        image_info.num_correspondences = elem.second;
        image_infos.push_back(image_info);
    }


    std::sort(
          image_infos.begin(), image_infos.end(),
          [](const ImageInfo& image_info1, const ImageInfo& image_info2)
            {
                return image_info1.num_correspondences > image_info2.num_correspondences;
            }
           );
    std::vector<image_t> image_ids;

    for(const ImageInfo& image_info : image_infos)
    {
        image_ids.push_back(image_info.image_id);
    }

    return image_ids;
}


bool Mapper::Initialize(image_t image_id1, image_t image_id2)
{
    std::cout << "---------------Initialize : " << image_id1 <<  " -- " << image_id2 << " ---------------------" << std::endl;
    // 增加注册的次数
    num_trial_registrations_[image_id1] += 1;
    num_trial_registrations_[image_id2] += 1;

    // 读取数据
    LoadImageData(image_id1);
    LoadImageData(image_id2);

    Image& image1 = images_[image_id1];
    Image& image2 = images_[image_id2];






    const std::vector<cv::Point2f>& pts1 = image1.Point2Ds();
    const std::vector<cv::Point2f>& pts2 = image2.Point2Ds();

    // 获取特征点匹配
    std::vector<cv::DMatch> matches = scene_graph_->FindCorrespondencesBetweenImages(image_id1, image_id2);


    // 获得对齐之后的特征点
    std::vector<cv::Point2f> aligned_pts1, aligned_pts2;
    FeatureUtils::GetAlignedPointsFromMatches(pts1, pts2, matches, aligned_pts1, aligned_pts2);

    // 恢复位姿
    cv::Mat inlier_mask;
    cv::Mat E = cv::findEssentialMat(aligned_pts1, aligned_pts2, K_, cv::RANSAC, 0.99, config_.max_2d_reprojection_error, inlier_mask);

    // 判断是否注册失败1
    if(cv::countNonZero(inlier_mask) < config_.initialize_pose_fail_threshold)
        return false;

    image1.R() = cv::Mat::eye(3, 3, CV_32F);
    image1.t() = cv::Mat::zeros(3, 1, CV_32F);

    recoverPose(E, aligned_pts1, aligned_pts2, K_, image2.R(), image2.t());

    image2.R().convertTo(image2.R(), CV_32F);
    image2.t().convertTo(image2.t(), CV_32F);


    const cv::Mat& R1 = image1.R();
    const cv::Mat& t1 = image1.t();
    const cv::Mat& R2 = image2.R();
    const cv::Mat& t2 = image2.t();

#ifdef DEBUG
    std::cout << " Initialize R " << std::endl;
    std::cout << R2 << std::endl;
    std::cout << " Initialize t " << std::endl;
    std::cout << t2 << std::endl;
#endif

    // 计算投影矩阵
    cv::Mat P1, P2;
    cv::hconcat(K_ * R1, K_ * t1, P1);
    cv::hconcat(K_ * R2, K_ * t2, P2);


    // 三角测量
    cv::Mat X;
    std::vector<cv::Point3f> point3ds;

    triangulatePoints(P1, P2, aligned_pts1, aligned_pts2, X);


    // 齐次坐标转为非齐次坐标
    convertPointsFromHomogeneous(X.t(), point3ds);


    // 过滤3D点
    PointFilter::RemoveWorldPtsByVisiable(point3ds, R1, t1, R2, t2, inlier_mask);
    PointFilter::RemoveWorldPtsByReprojectionError(point3ds, aligned_pts1, R1, t1, aligned_pts2, R2, t2, K_, inlier_mask, config_.max_3d_reprojection_error);
    PointFilter::RemoveWorldPtsByParallaxAngle(point3ds, R1, t1, R2, t2, inlier_mask, config_.min_parallax_degree);

//    // 判断是否注册失败2
    if(cv::countNonZero(inlier_mask)  < config_.initialize_triangulation_fail_threshold)
        return false;


    SetRegistered(image1);
    SetRegistered(image2);


    BuildInitialMap( image_id1, image_id2, point3ds, inlier_mask, matches);

    return true;
}


void Mapper::BuildInitialMap(const image_t& image_id1,
                             const image_t& image_id2,
                             const std::vector<cv::Point3f>& point3ds,
                             const cv::Mat& inlier_mask,
                             const std::vector<cv::DMatch>& matches)
{
    assert(inlier_mask.rows == point3ds.size());
    // 为3D点生成track
    std::unordered_map<point3D_t, MapPoint> init_map_points;

    for(int i = 0; i < inlier_mask.rows; ++i)
    {
        if(inlier_mask.at<uchar>(i, 0) == 0)
            continue;

        MapPoint map_point;
        map_point.SetPoint3DIdx(num_point3D_index_);
        map_point.SetPoint3D(point3ds[i]);


        point2D_t query_idx = matches[i].queryIdx;
        point2D_t train_idx = matches[i].trainIdx;

        map_point.AddElement(image_id1, query_idx);
        map_point.AddElement(image_id2, train_idx);


		// 当有新的map point产生时
		// 需要 1. 添加新的map point
		//	    2. 设置每张图片的2D-3D对应（已注册图片）
		//	    3. 设置每张图片能够看到的3D点(已注册图片 + 未注册图片)
        AddNewMapPoint(map_point, Color::COLOR_ORIGIN);
        SetImagePoint2D3DCorrespondence(image_id1, query_idx, num_point3D_index_);
        SetImagePoint2D3DCorrespondence(image_id2, train_idx, num_point3D_index_);
        const int inc = 1;
        UpdateImageVisable3DNum(map_point, inc);

        num_point3D_index_++;
    }


    std::cout << "initialize triangulate : "
              << image_id1 << " "
              << image_id2 << " ==> "
              << countNonZero(inlier_mask)<< " landmarks" << std::endl;

}
void Mapper::AddNewMapPoint(const MapPoint& map_point, Color color)
{
    map_points_[map_point.Point3DIdx()] = map_point;
    map_points_color_[map_point.Point3DIdx()] = color;
}

void Mapper::SetImagePoint2D3DCorrespondence(const image_t& image_id,
                                             const point2D_t& point2D_idx,
                                             const point3D_t& point3D_idx)
{
    images_[image_id].SetPoint2D3DCorrespondence(point2D_idx, point3D_idx);
}


void Mapper::SetRegistered(Image image)
{
    num_registered_ += 1;
    registered_[image.ImageId()] = true;
}

bool Mapper::IsRegistered(image_t image_id) const
{
    return (registered_.count(image_id) > 0) && registered_.at(image_id);
}


void Mapper::UpdateImageVisable3DNum(const MapPoint& map_point, const int& inc)
{
    assert(inc == 1 || inc == -1);

    for(const MapPointElement& element : map_point.Elements())
    {
        element.image_id;
        element.point2D_idx;

        const auto corrs = scene_graph_->FindCorrespondences(element.image_id, element.point2D_idx);

        for(const auto& corr : corrs)
        {
            corr.image_id;
            corr.point2D_idx;

            if(inc == 1)
                images_[corr.image_id].SetPointVisable(corr.point2D_idx);
            else
            {
                images_[corr.image_id].DisablePointVisable(corr.point2D_idx);
            }
        }

    }

}






std::vector<image_t> Mapper::FindNextImages() const
{
    struct ImageInfo
    {
        image_t image_id;
        size_t num_visable_point3Ds;
    };
    std::vector<ImageInfo> image_infos;
    std::vector<ImageInfo> image_infos2;

    for(const auto& ele : images_)
    {
        image_t image_id = ele.first;
        const Image& image = ele.second;

        // 跳过已经注册的图片
        if(IsRegistered(image_id))
            continue;

        size_t num_visable_point3D = image.NumVisablePoint3D();

        ImageInfo image_info;
        image_info.image_id = image_id;
        image_info.num_visable_point3Ds = num_visable_point3D;

        if(num_trial_registrations_.count(image_id) > 0)
        {
            image_infos2.push_back(image_info);
        }
        else
        {
            image_infos.push_back(image_info);
        }
    }

    std::sort(
          image_infos.begin(), image_infos.end(),
          [](const ImageInfo& image_info1, const ImageInfo& image_info2)
            {
                return image_info1.num_visable_point3Ds > image_info2.num_visable_point3Ds;
            }
           );

    std::sort(
          image_infos2.begin(), image_infos2.end(),
          [](const ImageInfo& image_info1, const ImageInfo& image_info2)
            {
                return image_info1.num_visable_point3Ds > image_info2.num_visable_point3Ds;
            }
           );

    std::vector<image_t> image_ids;

    for(const ImageInfo& image_info : image_infos)
    {
        image_ids.push_back(image_info.image_id);
    }

    for(const ImageInfo& image_info : image_infos2)
    {
        image_ids.push_back(image_info.image_id);
    }

    return image_ids;
}





bool Mapper::RegisterNextImage(image_t image_id)
{
    std::cout << std::endl;
    std::cout << "------------------------RegisterNextView : " << image_id << " -------------------------" << std::endl;
    // 增加注册的次数
    num_trial_registrations_[image_id] += 1;

    // 读取数据
    LoadImageData(image_id);

    Image& image = images_[image_id];


    std::vector<cv::Point2f> point2Ds;
    std::vector<cv::Point3f> point3Ds;
    std::vector<point2D_t> p2D_idx;
    std::vector<point3D_t> p3D_idx;
	// 得到2D-3D点对应
    Get2D3DCorrespondence(image, point2Ds, point3Ds, p2D_idx, p3D_idx);
    std::cout << "point2Ds.size() = " << point2Ds.size() << ", point3Ds.size() = " << point3Ds.size() << std::endl;


    if(point2Ds.size() < config_.register_2D_3D_correspodences_threshold)
        return false;

    // 求解位姿
    cv::Mat inlier_mask;
    cv::Mat rvec, tvec;
    /// PnP求解位姿
    /// 需要注意的是solvePnPRansac的mask输出
    /// inlier_idx = mask[i] 表示inlier_idx个点是内点，
    /// 此时的mask的值不再是0和1
    cv::solvePnPRansac(point3Ds, point2Ds, K_, cv::Mat(), rvec, tvec, true, 1000, 5.0, 0.999, inlier_mask);
    std::cout << "after solvePnPRansac, " <<
                 "point2Ds.size() = " << inlier_mask.rows <<
                 ", point3Ds.size() = " << inlier_mask.rows << std::endl << std::endl;


    if(inlier_mask.rows < config_.register_2D_3D_correspodences_threshold)
        return false;

    rvec.convertTo(rvec, CV_32F);
    tvec.convertTo(tvec, CV_32F);

    // 新图片的位姿
    cv::Rodrigues(rvec, image.R());
    image.t() = tvec;




#ifdef DEBUG
    std::cout << "next View R" << std::endl;
    std::cout << images_[image_id].R() << std::endl;
    std::cout << "next View t" << std::endl;
    std::cout << images_[image_id].t() << std::endl;
#endif


    for(int i = 0; i < inlier_mask.rows; ++i)
    {
        int inlier_idx = inlier_mask.at<int>(i, 0);

        point2D_t point2D_idx = p2D_idx[inlier_idx];
        point3D_t point3D_idx = p3D_idx[inlier_idx];

        // 这个assert报错, 因为一个2D点可能对应多个不同的3D点
//        assert(not image.IsPoint2DHasPoint3D(point2D_idx));
        if(image.IsPoint2DHasPoint3D(point2D_idx))
        {
            //std::cout << "Point2DHasPoint3D" << std::endl;
            continue;
        }


        UpdateMapPoint(image_id, point2D_idx, point3D_idx);
        SetImagePoint2D3DCorrespondence(image_id, point2D_idx, point3D_idx);

        const int inc = 1;
        UpdateImageVisable3DNum(map_points_[point3D_idx], inc);
    }

    SetRegistered(image);
    TriangulateImage(image_id);
    LocalBA(image_id);

}


void Mapper::Get2D3DCorrespondence(const Image& image,
                                   std::vector<cv::Point2f>& point2Ds,
                                   std::vector<cv::Point3f>& point3Ds,
                                   std::vector<point2D_t>& p2D_idx,
                                   std::vector<point3D_t>& p3D_idx)
{
    for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
    {

        // 得到该特征点的所有2D对应
        /// 注意
        /// 由于噪声的存在
        /// 虽然这些这些2D点都是同一个点的的对应点
        /// 但是可能导致该特征点对应着多个3D点
        std::vector<typename SceneGraph::Correspondence> corrs
                =  scene_graph_->FindCorrespondences(image.ImageId(), point2D_idx);
        for(auto& corr : corrs)
        {

            // 跨越多视图的2D-3D对应是非常有益处的，
            // 提高程序的稳定性， 健壮性
            if(!IsRegistered(corr.image_id))
                continue;

            bool has_point3D = images_[corr.image_id].IsPoint2DHasPoint3D(corr.point2D_idx);

            if(has_point3D)
            {

                point2D_t point3D_idx = images_[corr.image_id].GetPoint2D3DCorrespondence(corr.point2D_idx);

                p2D_idx.push_back(point2D_idx);
                p3D_idx.push_back(point3D_idx);

                const cv::Point2f& p2D = image.Point2D(point2D_idx);
                const cv::Point3f& p3D = map_points_[point3D_idx].Point3D();

                // 得到2D-3D对应
                point2Ds.push_back(p2D);
                point3Ds.push_back(p3D);

            }
        }
    }
}

void Mapper::UpdateMapPoint(const image_t &image_id,
                            const point2D_t &point2D_idx,
                            const point3D_t &point3D_idx)
{
    MapPoint& map_point = map_points_[point3D_idx];
    map_point.AddElement(image_id, point2D_idx);
    map_points_color_[point3D_idx] = Color::COLOR_GREEN;
}




void Mapper::TriangulateImage(const image_t &image_id)
{
    Timer timer;
    timer.Start();

    const Image& image = images_[image_id];

    SceneGraph::Correspondence cur_image;
    cur_image.image_id = image_id;

    int num_triangulation = 0;


    std::map<size_t, size_t> cnt;
    std::map<size_t, size_t> is_good_cnt;
    std::map<size_t, size_t> filtered_by_error_cnt;
    std::map<size_t, size_t> filtered_by_visable_cnt;

    for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++ point2D_idx)
    {
        // 如果已经有了3D点, 跳过
        if(image.IsPoint2DHasPoint3D(point2D_idx))
            continue;

        std::vector<SceneGraph::Correspondence> corrs =
                scene_graph_->FindCorrespondences(image_id, point2D_idx);

        if(corrs.size() == 0)
            continue;

        std::vector<SceneGraph::Correspondence> new_corrs;
        for(const auto& corr : corrs)
        {

            // 如果匹配点所在的图片没有注册, 跳过
            if(!IsRegistered(corr.image_id))
                continue;
            // 如果匹配点已经有了3D点, 跳过
            // 只对没有3D点的匹配点进行三角测量(这也也会造成track的分离)
            if(images_[corr.image_id].IsPoint2DHasPoint3D(corr.point2D_idx))
                continue;

            new_corrs.push_back(corr);
        }

        if(new_corrs.size() == 0)
            continue;

        cnt[new_corrs.size()] += 1;

        // 多视图三角测量
        std::vector<cv::Mat> R;
        std::vector<cv::Mat> t;
        std::vector<cv::Point2f> point2Ds;

        cur_image.point2D_idx = point2D_idx;
        new_corrs.push_back(cur_image);

        for(const auto& new_corr : new_corrs)
        {
            R.push_back(images_[new_corr.image_id].R());
            t.push_back(images_[new_corr.image_id].t());

            point2Ds.push_back(images_[new_corr.image_id].Point2D(new_corr.point2D_idx));
        }
        cv::Point3f point3D = TriangulateMultiViewPoint(K_, R, t, point2Ds);

        bool is_good = true;
        for(size_t i = 0; i < R.size(); ++i)
        {

           float error = PointFilter::CalculateReprojectionError(point3D, point2Ds[i], R[i], t[i], K_);

           if(error > 3.0)
           {
               is_good = false;
               filtered_by_error_cnt[new_corrs.size()] += 1;
               break;
           }



           if(!PointFilter::HasPositiveDepth(point3D, R[i], t[i]))
           {
               is_good = false;
               filtered_by_visable_cnt[new_corrs.size()] += 1;
               break;
           }

        }

        if(is_good)
        {
            is_good = false;
            for(size_t i = 0; i < R.size(); ++i)
            {
                for(size_t j = 0; j < i; ++j)
                {
                    // 只要有一个满足角度要求, 那么就不进行过滤
                    if(not PointFilter::RemoveWorldPtsByParallaxAngle(point3D, R[i], t[i], R[j], t[j]))
                    {
                        is_good = true;
                        break;
                    }

                }
            }
        }

        if(is_good)
        {
            is_good_cnt[new_corrs.size()] += 1;

            // 生成Track
            MapPoint map_point;
            map_point.SetPoint3D(point3D);
            map_point.SetPoint3DIdx(num_point3D_index_);
            for(const auto& new_corr : new_corrs)
            {
                new_corr.image_id;
                new_corr.point2D_idx;

                map_point.AddElement(new_corr.image_id, new_corr.point2D_idx);

                SetImagePoint2D3DCorrespondence(new_corr.image_id, new_corr.point2D_idx, num_point3D_index_);

            }

            AddNewMapPoint(map_point, Color::COLOR_RED);
            const int inc = 1;
            UpdateImageVisable3DNum(map_point, inc);
            num_triangulation += 1;
            num_point3D_index_ += 1;

        }

    }
    std::cout << "Triangulate more points: "
              << num_triangulation << " point3ds.  "
              << std::endl;


    for(const auto ele : cnt)
    {
        size_t track_size = ele.first + 1;
        size_t track_num = ele.second;
        size_t is_good_track_num = is_good_cnt[track_size];
        size_t num_track_filtered_by_error = filtered_by_error_cnt[track_size];
        size_t num_track_filtered_by_visable = filtered_by_visable_cnt[track_size];

        std::cout << "\t Track Size = " << track_size << std::endl
                  << "\t\t num : " << track_num << std::endl
                  << "\t\t is good track  : " << is_good_track_num << std::endl
                  << "\t\t filtered by error : " << num_track_filtered_by_error << std::endl
                  << "\t\t filtered by visable : " << num_track_filtered_by_visable << std::endl << std::endl;

    }
    std::cout << "\t ";
    timer.PrintSeconds();
    std::cout << std::endl;

}








void Mapper::Retriangulate()
{
    // 对于under-reconstruction的图片进行re-triangulate
    for(const auto& ele : images_)
    {
        image_t image_id = ele.first;
        const Image& image = ele.second;

        const float kReTriangulationRatio = 0.2;

        const double triangulation_ratio =
                static_cast<double>(image.NumPoint2D3DCorrespondence()) / image.NumPoints2D();

        if(not IsRegistered(image_id))
            continue;

        if(triangulation_ratio >= kReTriangulationRatio)
            continue;

        const int kMaxReTriangulate = 5;
        if(num_re_triangulate_[image_id] > kMaxReTriangulate)
            continue;
        num_re_triangulate_[image_id] += 1;
        RetriangulateImage(image_id);


    }
}

void Mapper::RetriangulateImage(const image_t &image_id)
{
    std::cout << "Re-triangulate " << std::endl;
    TriangulateImage(image_id);
}




void Mapper::FilterAllMapPoints()
{
    Timer timer;
    timer.Start();

    size_t outlier_num = 0;
    size_t total_num = map_points_.size();

    is_deleted_.clear();


    // 删除不合格的track
    for(const auto& ele : map_points_)
    {
        point3D_t point3D_idx = ele.first;
        outlier_num += FilterMapPoint(point3D_idx);
    }
    DeletedMapPoints();

    std::cout << "Global FilterAllMapPoints " << std::endl;;
    std::cout << "\t remove " << outlier_num << " outliers outof " << total_num  << std::endl;
    std::cout << "\t ";
    timer.PrintSeconds();
    std::cout << std::endl;

}

void Mapper::FilterMapPointsInImages(const std::vector<image_t> &image_ids)
{

    std::vector<point3D_t> point3D_ids;
    std::unordered_set<point3D_t> vis;
    for(image_t image_id : image_ids)
    {
        Image& image = images_[image_id];
        for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++ point2D_idx)
        {
            if(!image.IsPoint2DHasPoint3D(point2D_idx))
                continue;

            point3D_t point3D_idx = image.GetPoint2D3DCorrespondence(point2D_idx);

            if(vis.count(point3D_idx) > 0)
                continue;
            vis.insert(point3D_idx);
            point3D_ids.push_back(point3D_idx);
        }
    }

    FilterMapPoints(point3D_ids);

}

void Mapper::FilterMapPoints(std::vector<point3D_t> &point3D_ids)
{

    Timer timer;
    timer.Start();

    size_t outlier_num = 0;
    size_t total_num = point3D_ids.size();

    is_deleted_.clear();


    // 删除不合格的track
    for(const auto& point3D_idx : point3D_ids)
    {
        outlier_num += FilterMapPoint(point3D_idx);
    }

    DeletedMapPoints();

    std::cout << "Local FilterMapPoints " << std::endl;
    std::cout << "\t remove " << outlier_num << " outliers outof " << total_num  << std::endl;
    std::cout << "\t ";
    timer.PrintSeconds();
    std::cout << std::endl;

}

bool Mapper::FilterMapPoint(point3D_t point3D_idx)
{
    // 已经被过滤掉了
    if(not HasMapPoint(point3D_idx))
        return true;


    MapPoint& map_point = map_points_[point3D_idx];
    std::vector<cv::Point2f> pts;
    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> ts;
    for(MapPointElement& element : map_point.Elements())
    {

        image_t image_id  = element.image_id;
        point2D_t point2D_idx = element.point2D_idx;

        Image& image = images_[image_id];
        pts.push_back(image.Point2D(point2D_idx));
        Rs.push_back(image.R());
        ts.push_back(image.t());

    }
    cv::Mat inlier_mask = cv::Mat::ones(map_point.Elements().size(), 1, CV_8U);
    bool is_filter = TrackFilter::RemoveTrack(map_point.Point3D(), pts, Rs, ts, K_, inlier_mask, config_.filtered_reprojection_error);
    if(is_filter)
    {

        for(MapPointElement& element : map_point.Elements())
        {

            image_t image_id  = element.image_id;
            point2D_t point2D_idx = element.point2D_idx;

            Image& image = images_[image_id];

            // 设置对应图像的2D点不能够看到3D点

            image.DisablePoint2D3DCorrespondence(point2D_idx);
            image.DisablePointVisable(point2D_idx);

        }
        // 待删除
        is_deleted_[point3D_idx] = true;
    }

    return is_filter;
}

void Mapper::DeletedMapPoints()
{
    for(const auto& ele : is_deleted_)
    {
        if(ele.second)
        {
            const int inc = -1;
            UpdateImageVisable3DNum(map_points_[ele.first], inc);

            map_points_.erase(ele.first);
            map_points_color_.erase(ele.first);

        }
    }
}




void Mapper::CompleteAllMapPoints()
{
    Timer timer;
    timer.Start();


    size_t num_completed = 0;
    for(const auto& map_point : map_points_)
    {
        num_completed += CompleteMapPoint(map_point.first);
    }
    std::cout << "Global CompleteAllMapPoints " << std::endl;
    std::cout << "\t total complete : " << num_completed << std::endl;
    std::cout << "\t ";
    timer.PrintSeconds();
    std::cout << std::endl;

}

void Mapper::CompleteMapPoints(const std::vector<point3D_t> &point3D_ids)
{
    Timer timer;
    timer.Start();


    size_t num_completed = 0;
    for(const auto& point3D_idx : point3D_ids)
    {
        if(map_points_.count(point3D_idx) == 0)
            continue;

        num_completed += CompleteMapPoint(point3D_idx);
    }
    std::cout <<"Local CompleteMapPoints " << std::endl;
    std::cout << "\t total complete : " << num_completed << std::endl;
    std::cout << "\t ";
    timer.PrintSeconds();
    std::cout << std::endl;

}

size_t Mapper::CompleteMapPoint(point3D_t point3D_idx)
{


    // 对point3D_idx 这个3D点所在的track进行补全
    size_t num_completed = 0;


    cv::Point3f point3D = map_points_[point3D_idx].Point3D();

    std::vector<MapPointElement> queue;
    queue.reserve(map_points_[point3D_idx].Length());

    for(const MapPointElement& element : map_points_[point3D_idx].Elements())
    {
        queue.emplace_back(element.image_id, element.point2D_idx);
    }

    const int max_transitivity = 5;

    for(int transitivity = 0; transitivity < max_transitivity; ++ transitivity)
    {
        if(queue.empty())
            break;

        const auto prev_queue = queue;
        queue.clear();

        for(const MapPointElement& element : prev_queue)
        {
            const std::vector<SceneGraph::Correspondence>& corrs
                    = scene_graph_->FindCorrespondences(element.image_id, element.point2D_idx);

            for(const auto corr : corrs)
            {
                if(corr.image_id == element.image_id)
                    continue;

                if(!IsRegistered(corr.image_id))
                    continue;

                Image& image = images_[corr.image_id];

                if(image.IsPoint2DHasPoint3D(corr.point2D_idx))
                    continue;


                cv::Point2f point2D = image.Point2D(corr.point2D_idx);
                if(! PointFilter::HasPositiveDepth(point3D, image.R(), image.t()))
                    continue;

                const double error = PointFilter::CalculateReprojectionError(point3D, point2D, image.R(), image.t(), K_);

                if(error >= config_.complete_reprojection_error)
                    continue;


                UpdateMapPoint(corr.image_id, corr.point2D_idx, point3D_idx);
                SetImagePoint2D3DCorrespondence(corr.image_id, corr.point2D_idx, point3D_idx);

                queue.emplace_back(corr.image_id, corr.point2D_idx);

                num_completed += 1;
            }
        }
    }

    const int inc = 1;
    UpdateImageVisable3DNum(map_points_[point3D_idx], inc);

    return num_completed;
}


void Mapper::CompleteImages(const std::vector<image_t> &image_ids)
{
    for(image_t image_id : image_ids)
        CompleteImage(image_id);
}
void Mapper::CompleteImage(const image_t &image_id)
{
    RetriangulateImage(image_id);
}




void Mapper::MergeAllMapPoints()
{
    Timer timer;
    timer.Start();


    merge_trials_.clear();
    is_deleted_.clear();

    size_t num_merged = 0;

    for(const auto& map_point : map_points_)
    {
        num_merged += MergeMapPoint(map_point.first);
    }

    DeletedMapPoints();

    std::cout << "Global MergeAllMapPoints " << std::endl;
    std::cout << "\t total merge : " << num_merged << std::endl;
    std::cout << "\t ";
    timer.PrintSeconds();
    std::cout << std::endl;

}

void Mapper::MergeMapPoints(const std::vector<point3D_t> &point3D_ids)
{
    Timer timer;
    timer.Start();


    merge_trials_.clear();
    is_deleted_.clear();

    size_t num_merged = 0;

    for(const auto& point3D_idx : point3D_ids)
    {
        num_merged += MergeMapPoint(point3D_idx);
    }

    DeletedMapPoints();
    std::cout << "Local MergeMapPoints " << std::endl;
    std::cout << "\t total merge : " << num_merged << std::endl;
    std::cout << "\t ";
    timer.PrintSeconds();
    std::cout << std::endl;

}

size_t Mapper::MergeMapPoint(point3D_t point3D_idx)
{
    if(is_deleted_.count(point3D_idx) > 0)
        return 0;
    if(not HasMapPoint(point3D_idx))
        return 0;

    size_t num_merged = 0;

    const MapPoint& map_point = map_points_[point3D_idx];

    for(const MapPointElement& element : map_point.Elements())
    {
        const std::vector<SceneGraph::Correspondence>& corrs
                = scene_graph_->FindCorrespondences(element.image_id, element.point2D_idx);

        for(const auto& corr : corrs)
        {
            if(!registered_[corr.image_id])
                continue;


            if(is_deleted_.count(point3D_idx) > 0)
                continue;

            if(!images_[corr.image_id].IsPoint2DHasPoint3D(corr.point2D_idx))
                continue;

            point3D_t other_point3D_idx = images_[corr.image_id].GetPoint2D3DCorrespondence(corr.point2D_idx);

            if(other_point3D_idx == point3D_idx
                || merge_trials_[point3D_idx].count(other_point3D_idx) > 0)
                continue;


            // 合并两个3D点

            cv::Point3f point3D1 = map_points_[point3D_idx].Point3D();
            cv::Point3f point3D2 = map_points_[other_point3D_idx].Point3D();

            float w1 = map_points_[point3D_idx].Length();
            float w2 = map_points_[other_point3D_idx].Length();

            float x = (w1 * point3D1.x + w2 * point3D2.x) / (w1 + w2);
            float y = (w1 * point3D1.y + w2 * point3D2.y) / (w1 + w2);
            float z = (w1 * point3D1.z + w2 * point3D2.z) / (w1 + w2);

            cv::Point3f merged_point3D(x, y, z);

            size_t num_inliers = 0;
            for(const MapPoint* map_point : {&map_points_[point3D_idx], & map_points_[other_point3D_idx]})
            {
                for(const MapPointElement& element : map_point->Elements())
                {
                    element.image_id;
                    element.point2D_idx;

                    const Image& image = images_[element.image_id];
                    cv::Point2f point2D = image.Point2D(element.point2D_idx);

                    bool is_inlier = PointFilter::HasPositiveDepth(merged_point3D, image.R(), image.t());
                    is_inlier &= PointFilter::CalculateReprojectionError(merged_point3D, point2D, image.R(), image.t(), K_)
                                 < config_.merge_reprojection_error;
                    if(is_inlier)
                        num_inliers += 1;
                     else
                        break;

                }
            }

            // 如果合并之后都是内点, 那么进行合并
            if(num_inliers == map_points_[point3D_idx].Length() + map_points_[other_point3D_idx].Length())
            {
                num_merged += num_inliers;
                point3D_t merged_point3D_idx = MergePoint3D(point3D_idx, other_point3D_idx, merged_point3D);
                return num_merged + MergeMapPoint(merged_point3D_idx);
            }

        }

    }

    return num_merged;
}

point3D_t Mapper::MergePoint3D(point3D_t point3D_idx1, point3D_t point3D_idx2, cv::Point3f merged_point3D)
{
    MapPoint map_point;
    map_point.SetPoint3D(merged_point3D);
    map_point.SetPoint3DIdx(num_point3D_index_);
    map_point.AddElements(map_points_[point3D_idx1].Elements());
    map_point.AddElements(map_points_[point3D_idx2].Elements());

    for(const MapPointElement& element : map_point.Elements())
    {
        element.image_id;
        element.point2D_idx;

        Image& image = images_[element.image_id];
        image.SetPoint2D3DCorrespondence(element.point2D_idx, map_point.Point3DIdx());
    }


    is_deleted_[point3D_idx1]  = true;
    is_deleted_[point3D_idx2]  = true;

    AddNewMapPoint(map_point, map_points_color_[point3D_idx1]);


    const int inc = 1;
    UpdateImageVisable3DNum(map_points_[map_point.Point3DIdx()], inc);


    num_point3D_index_ += 1;


    return map_point.Point3DIdx();
}



void Mapper::GetLocalBAData(image_t image_id, BAData &ba_data)
{
    Timer timer;
    timer.Start();

    /**
        通过当前图像的特征点所在的Track（或通过当前图像的特征点的匹配关系）
        找到所有与当前图像有关系的图像
        遍历这些图像的Track， 生成优化关系
      */
    // 为什么const Image& image 通不过编译？
    Image& image = images_[image_id];

    std::unordered_map<image_t, size_t> num_shared_observations;
    std::unordered_set<image_t> related_images;

    // 根据图片之间3D点的共视关系, 找到与当前图片相关的图片
    for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
    {
        if(!image.IsPoint2DHasPoint3D(point2D_idx))
            continue;

        point3D_t point3D_idx =  image.GetPoint2D3DCorrespondence(point2D_idx);

        const MapPoint& map_point = map_points_[point3D_idx];

        for(const MapPointElement& element : map_point.Elements())
        {
            // 与当前图像有关系的，已经注册了的图像
            element.image_id;
//            related_images.insert(element.image_id);
            num_shared_observations[element.image_id] += 1;

        }
    }

    // 最大只保留kMaxRelatedImageNum张与之相关的图片
    const size_t kMaxRelatedImageNum = 10;

    std::vector<std::pair<image_t, size_t>> local_bundle;

    for(const auto& ele : num_shared_observations)
    {
        local_bundle.emplace_back(ele.first, ele.second);
    }

    const size_t num_eff_images = std::min(kMaxRelatedImageNum, local_bundle.size());

    // 按照共视3D点的数量(从大到小),对图像进行排序
    std::partial_sort(local_bundle.begin(), local_bundle.begin() + num_eff_images,
                      local_bundle.end(),
                      [](const std::pair<image_t, size_t>& image1,
                         const std::pair<image_t, size_t>& image2)
                      {
                            return image1.second > image2.second;
                      });

    for(size_t i = 0; i < num_eff_images; ++i)
    {
        related_images.insert(local_bundle[i].first);
    }
    related_images.insert(image_id);


    std::unordered_set<point3D_t> vis_map_points;
    std::vector<Landmark> landmarks;
    std::unordered_map<image_t, CameraPose> camera_poses;
    std::unordered_map<image_t, std::unordered_set<point2D_t> > vis_point2Ds;

    std::cout << "related_images size : " << related_images.size() << std::endl;
    for(image_t related_image_id : related_images)
    {

        Image& related_image = images_[related_image_id];
        for(point2D_t point2D_idx = 0; point2D_idx < related_image.NumPoints2D(); ++point2D_idx)
        {
            if(!related_image.IsPoint2DHasPoint3D(point2D_idx))
                continue;

            point3D_t point3D_idx =  related_image.GetPoint2D3DCorrespondence(point2D_idx);


            const MapPoint& map_point = map_points_[point3D_idx];


            vis_map_points.insert(map_point.Point3DIdx());

        }
    }




    for(point3D_t point3D_idx : vis_map_points)
    {


        const MapPoint& map_point = map_points_[point3D_idx];

        cv::Point3f point3D = map_point.Point3D();
        std::vector<Measurement> measurements;


        for(const MapPointElement& element : map_point.Elements())
        {

            cv::Point2f pt = images_[element.image_id].Point2D(element.point2D_idx);
            vis_point2Ds[element.image_id].insert(element.point2D_idx);
            measurements.emplace_back(element.image_id, pt);

            if(camera_poses.count(element.image_id) == 0)
            {
                cv::Mat R = images_[element.image_id].R().clone();
                cv::Mat t = images_[element.image_id].t().clone();
                CameraPose camera_pose(element.image_id, R, t);
                camera_poses[element.image_id] = camera_pose;
            }
        }
        landmarks.emplace_back(point3D_idx, point3D, measurements);
    }


    ba_data.K = K_.clone();
    ba_data.landmarks = landmarks;
    ba_data.camera_poses = camera_poses;

    std::cout << "GetLocalBAData ";
    timer.PrintSeconds();
    std::cout <<"ba_data.landmarks.size() : " <<  ba_data.landmarks.size() << std::endl;
    std::cout <<"ba_data.camera_poses.size() : " << ba_data.camera_poses.size() << std::endl;
}

void Mapper::GetLocalBAData2(image_t image_id, BAData &ba_data)
{
    Timer timer;
    timer.Start();

    /**
        通过当前图像的特征点所在的Track（或通过当前图像的特征点的匹配关系）
        找到所有与当前图像有关系的图像
        遍历这些图像的Track， 生成优化关系
      */
    // 为什么const Image& image 通不过编译？
    Image& image = images_[image_id];

    std::unordered_map<image_t, size_t> num_shared_observations;
    std::unordered_set<image_t> related_images;

    // 根据图片之间3D点的共视关系, 找到与当前图片相关的图片
    for(point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx)
    {
        if(!image.IsPoint2DHasPoint3D(point2D_idx))
            continue;

        point3D_t point3D_idx =  image.GetPoint2D3DCorrespondence(point2D_idx);

        const MapPoint& map_point = map_points_[point3D_idx];

        for(const MapPointElement& element : map_point.Elements())
        {
            // 与当前图像有关系的，已经注册了的图像
            element.image_id;
//            related_images.insert(element.image_id);
            num_shared_observations[element.image_id] += 1;

        }
    }

    // 最大只保留kMaxRelatedImageNum张与之相关的图片
    const size_t kMaxRelatedImageNum = 10;

    std::vector<std::pair<image_t, size_t>> local_bundle;

    for(const auto& ele : num_shared_observations)
    {
        local_bundle.emplace_back(ele.first, ele.second);
    }

    const size_t num_eff_images = std::min(kMaxRelatedImageNum, local_bundle.size());

    // 按照共视3D点的数量(从大到小),对图像进行排序
    std::partial_sort(local_bundle.begin(), local_bundle.begin() + num_eff_images,
                      local_bundle.end(),
                      [](const std::pair<image_t, size_t>& image1,
                         const std::pair<image_t, size_t>& image2)
                      {
                            return image1.second > image2.second;
                      });

    for(size_t i = 0; i < num_eff_images; ++i)
    {
        related_images.insert(local_bundle[i].first);
    }
    related_images.insert(image_id);


    std::unordered_set<point3D_t> vis_map_points;
    std::vector<Landmark> landmarks;
    std::unordered_map<image_t, CameraPose> camera_poses;
    std::unordered_map<image_t, std::unordered_set<point2D_t> > vis_point2Ds;

    std::cout << "related_images size : " << related_images.size() << std::endl;


    // 遍历相关图片的3D点, 这些3D点都是要进行优化的
    for(image_t related_image_id : related_images)
    {

        Image& related_image = images_[related_image_id];
        for(point2D_t point2D_idx = 0; point2D_idx < related_image.NumPoints2D(); ++point2D_idx)
        {
            if(!related_image.IsPoint2DHasPoint3D(point2D_idx))
                continue;

            point3D_t point3D_idx =  related_image.GetPoint2D3DCorrespondence(point2D_idx);


            const MapPoint& map_point = map_points_[point3D_idx];


            vis_map_points.insert(map_point.Point3DIdx());

        }
    }



    for(point3D_t point3D_idx : vis_map_points)
    {


        const MapPoint& map_point = map_points_[point3D_idx];

        cv::Point3f point3D = map_point.Point3D();
        std::vector<Measurement> measurements;

        for(const MapPointElement& element : map_point.Elements())
        {

            if(related_images.count(element.image_id) == 0)
                continue;

            cv::Point2f pt = images_[element.image_id].Point2D(element.point2D_idx);
            vis_point2Ds[element.image_id].insert(element.point2D_idx);
            measurements.emplace_back(element.image_id, pt);

            // 我觉得可以把相关图片的加到优化中
            // 而把不相关的图片加到约束中
            if( camera_poses.count(element.image_id) == 0)
            {
                cv::Mat R = images_[element.image_id].R().clone();
                cv::Mat t = images_[element.image_id].t().clone();
                CameraPose camera_pose(element.image_id, R, t);
                camera_poses[element.image_id] = camera_pose;
            }
        }
        landmarks.emplace_back(point3D_idx, point3D, measurements);
    }


    ba_data.K = K_.clone();
    ba_data.landmarks = landmarks;
    ba_data.camera_poses = camera_poses;

    std::cout << "GetLocalBAData ";
    timer.PrintSeconds();
    std::cout <<"ba_data.landmarks.size() : " <<  ba_data.landmarks.size() << std::endl;
    std::cout <<"ba_data.camera_poses.size() : " << ba_data.camera_poses.size() << std::endl;
}

void Mapper::LocalBA(image_t image_id)
{
    BAData ba_data;
    GetLocalBAData2(image_id, ba_data);

    CeresBA::Adjust(ba_data);

    UpdateFromBAData(ba_data);

    std::vector<image_t> image_ids;
    for(const auto& ele : ba_data.camera_poses)
    {
        image_t image_id = ele.first;
        image_ids.push_back(image_id);
    }

    std::vector<point3D_t> point3D_ids;
    for(const auto& landmark : ba_data.landmarks)
    {
        point3D_ids.push_back(landmark.point3D_idx);
    }

    FilterMapPointsInImages(image_ids);
//    FilterMapPoints(point3D_ids);

    MergeMapPoints(point3D_ids);
    CompleteMapPoints(point3D_ids);
    CompleteImages(image_ids);


}


void Mapper::GetGlobalBAData(BAData& ba_data)
{
    Timer timer;
    timer.Start();

    /// 遍历所有的Track
    /// 遍历Track中的2D点和相机， 生成优化关系
    std::vector<Landmark> landmarks;
    std::unordered_map<image_t, CameraPose> camera_poses;

    for(const auto& ele : map_points_)
    {

        point3D_t point3D_idx = ele.first;

        const MapPoint& map_point = ele.second;
        cv::Point3f point3D = map_point.Point3D();
        std::vector<Measurement> measurements;

        for(const MapPointElement& element : map_point.Elements())
        {


            cv::Point2f pt = images_[element.image_id].Point2D(element.point2D_idx);
            measurements.emplace_back(element.image_id, pt);

            if(camera_poses.count(element.image_id) == 0)
            {
                cv::Mat R = images_[element.image_id].R().clone();
                cv::Mat t = images_[element.image_id].t().clone();
                CameraPose camera_pose(element.image_id, R, t);
                camera_poses[element.image_id] = camera_pose;
            }

        }
        landmarks.emplace_back(point3D_idx, point3D, measurements);
    }
    ba_data.K = K_.clone();
    ba_data.landmarks = landmarks;
    ba_data.camera_poses = camera_poses;

    std::cout << "GetGlobalBAData ";
    timer.PrintSeconds();

}

void Mapper::GlobalBA()
{
    BAData ba_data;
    GetGlobalBAData(ba_data);
    CeresBA::Adjust(ba_data);
    UpdateFromBAData(ba_data);

}

void Mapper::UpdateFromBAData(BAData &ba_data)
{
    for(auto& ele : ba_data.camera_poses)
    {
        image_t image_id = ele.first;
        CameraPose camera_pose = ele.second;
        images_[image_id].R() =  camera_pose.R;
        images_[image_id].t() = camera_pose.t;

    }

    for(Landmark& landmark : ba_data.landmarks)
    {
        point3D_t point3D_idx = landmark.point3D_idx;
        cv::Point3f point3D = landmark.point3D;

        map_points_[point3D_idx].Point3D() = point3D;
    }
}







bool Mapper::HasMapPoint(const point3D_t &point3D_idx)
{
    return map_points_.count(point3D_idx) != 0;
}

bool Mapper::IsNeedUndistortFeature()
{
    const float kEps = 1e-5;
    if(fabs(config_.k1- (0.0)) < kEps &&
       fabs(config_.k2- (0.0)) < kEps &&
       fabs(config_.p1- (0.0)) < kEps &&
       fabs(config_.p2- (0.0)) < kEps)
    {
        return false;
    }
    return true;
}


void Mapper::UndistortFeature(image_t image_id)
{

    std::vector<cv::Point2f>& point2Ds = images_[image_id].Point2Ds();

    cv::Mat mat(point2Ds.size(), 2, CV_32F);
    for(size_t i = 0; i < point2Ds.size(); ++i)
    {
        mat.at<float>(i, 0) = point2Ds[i].x;
        mat.at<float>(i, 1) = point2Ds[i].y;
    }
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, K_, dist_coef_, cv::Mat(), K_);
    mat = mat.reshape(1);

    for(size_t i = 0; i < point2Ds.size(); ++i)
    {
        cv::Point2f point2D(mat.at<float>(i, 0), mat.at<float>(i, 1));

        point2Ds[i] = point2D;

    }
}


float Mapper::GetMapPointError(const MapPoint& map_point)
{
    float error = 0;
    for(const MapPointElement& element : map_point.Elements())
    {
        image_t image_id = element.image_id;
        point2D_t point2D_idx = element.point2D_idx;

        cv::Point2f point2D = images_[image_id].Point2D(point2D_idx);
        cv::Point3f point3D = map_point.Point3D();

        cv::Mat R = images_[image_id].R();
        cv::Mat t = images_[image_id].t();


        error += PointFilter::CalculateReprojectionError(point3D, point2D, R, t, K_);

    }
    return error;
}
float Mapper::GetAveMapPointsError()
{
    float error = 0;
    for(const auto& ele : map_points_)
    {
        error += GetMapPointError(ele.second);
    }
    error /= map_points_.size();
    return error;
}
