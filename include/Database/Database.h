#ifndef __DATABASE_H__
#define __DATABASE_H__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ext/SQLite/sqlite3.h>

#include "Common/Types.h"

namespace MonocularSfM
{




class Database
{
public:
    struct Image
    {
        image_t id;
        std::string name;
    };

    const static int kSchemaVersion = 1;

    Database();
    ~Database();
    void Open(const std::string& path);
    void Close();

    void BeginTransaction() const;
    void EndTransaction() const;


    bool ExistImageById(const image_t image_id) const;
    bool ExistImageByName(const std::string name) const;
    bool ExistKeyPoints(const image_t image_id) const;
    bool ExistKeyPointsColor(const image_t image_id) const;
    bool ExistDescriptors(const image_t image_id) const;
    bool ExistMatches(const image_pair_t pair_id) const;
    bool ExistMatches(const image_t image_id1, const image_t image_id2) const;


    size_t NumImages() const;
    size_t NumKeyPoints(const image_t image_id) const;
    size_t NumKeyPointsColor(const image_t image_id) const;
    size_t NumDescriptors(const image_t image_id) const;
    size_t NumMatches(const image_pair_t pair_id) const;
    size_t NumMatches(const image_t image_id1, const image_t image_id2) const;


    Image ReadImageById(const image_t image_id) const;
    Image ReadImageByName(const std::string name) const;
    std::vector<Image> ReadAllImages() const;
    std::vector<cv::KeyPoint> ReadKeyPoints(const image_t image_id) const;
    std::vector<cv::Vec3b> ReadKeyPointsColor(const image_t image_id) const;
    cv::Mat ReadDescriptors(const image_t image_id) const;
    std::vector<cv::DMatch> ReadMatches(const image_pair_t pair_id) const;
    std::vector<cv::DMatch> ReadMatches(const image_t image_id1, const image_t image_id2) const;
    std::vector<std::pair<image_pair_t, std::vector<cv::DMatch>>> ReadAllMatches() const;


    image_t WriteImage(const Image& image, const bool use_image_id = false) const;
    void WriteKeyPoints(const image_t image_id, const std::vector<cv::KeyPoint>& keypoints) const;
    void WriteKeyPointsColor(const image_t image_id, const std::vector<cv::Vec3b>& keypoints) const;
    void WriteDescriptors(const image_t image_id, const cv::Mat& descriptors) const;
    void WriteMatches(const image_t image_id1, const image_t image_id2, const std::vector<cv::DMatch>& matches) const;


    static image_pair_t ImagePairToPairId(const image_t image_id1, const image_t image_id2);
    static void PairIdToImagePair(const image_pair_t pair_id, image_t* image_id1, image_t* image_id2);
    static bool SwapImagePair(const image_t image_id1, const image_t image_id2);

private:


    // 创建数据库表
    void CreateTables() const;
    void CreateImageTable() const;
    void CreateKeyPointsTable() const;
    void CreateKeyPointsColorTable() const;
    void CreateDescriptorsTable() const;
    void CreateMatchesTable() const;

    // 创建和销毁SQL语句
    void PrepareSQLStatements();
    void FinalizeSQLStatements();


    void UpdateSchema() const;
    bool ExistRowId(sqlite3_stmt* sql_stmt, const size_t row_id) const;
    bool ExistRowString(sqlite3_stmt* sql_stmt, const std::string& row_entry) const;
    size_t CountRows(const std::string& table) const;


private:
    sqlite3* database_;
    std::vector<sqlite3_stmt*> sql_stmts_;

    // exists_*
    sqlite3_stmt* sql_stmt_exists_image_id_;
    sqlite3_stmt* sql_stmt_exists_image_name_;
    sqlite3_stmt* sql_stmt_exists_keypoints_;
    sqlite3_stmt* sql_stmt_exists_keypoints_color_;
    sqlite3_stmt* sql_stmt_exists_descriptors_;
    sqlite3_stmt* sql_stmt_exists_matches_;

    //read_*
    sqlite3_stmt* sql_stmt_read_image_id_;
    sqlite3_stmt* sql_stmt_read_image_name_;
    sqlite3_stmt* sql_stmt_read_images_;
    sqlite3_stmt* sql_stmt_read_keypoints_;
    sqlite3_stmt* sql_stmt_read_keypoints_num_;
    sqlite3_stmt* sql_stmt_read_keypoints_color_;
    sqlite3_stmt* sql_stmt_read_keypoints_color_num_;
    sqlite3_stmt* sql_stmt_read_descriptors_;
    sqlite3_stmt* sql_stmt_read_descriptors_num_;
    sqlite3_stmt* sql_stmt_read_matches_;
    sqlite3_stmt* sql_stmt_read_matches_num_;
    sqlite3_stmt* sql_stmt_read_matches_all_;


    //add_*
    sqlite3_stmt* sql_stmt_add_image_;
    sqlite3_stmt* sql_stmt_add_keypoints_;
    sqlite3_stmt* sql_stmt_add_keypoints_color_;
    sqlite3_stmt* sql_stmt_add_descriptors_;
    sqlite3_stmt* sql_stmt_add_matches_;



};










} // namespace Undifine


#endif // __DATABASE_H__
