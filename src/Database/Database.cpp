#include "Database/Database.h"

using namespace MonocularSfM;


const size_t kMaxNumImages = 1000000;

inline int SQLite3CallHelper(const int result_code, const std::string& filename, const int line_number)
{
    switch(result_code)
    {
    case SQLITE_OK:
    case SQLITE_ROW:
    case SQLITE_DONE:
        return result_code;
    default:
        fprintf(stderr, "SQLite error [%s, line %i]: %s\n",
                filename.c_str(),line_number, sqlite3_errstr(result_code));
        exit(EXIT_FAILURE);

    }
}

#define SQLITE3_CALL(func) SQLite3CallHelper(func, __FILE__, __LINE__)

#define SQLITE3_EXEC(database, sql, callback)                                   \
        {                                                                       \
            char*err_msg = nullptr;                                             \
            int rc = sqlite3_exec(database, sql, callback, nullptr, &err_msg);  \
            if(rc != SQLITE_OK)                                                 \
            {                                                                   \
                fprintf(stderr, "SQLite error [%s, line %i]: %s\n",             \
                        __FILE__, __LINE__, err_msg);                           \
                sqlite3_free(err_msg);                                          \
            }                                                                   \
        }




template<typename T>
class Blob
{
public:
    typedef T value_type;
    typedef size_t Index;
    size_t rows;
    size_t cols;
    Blob()
    {
        rows = 0;
        cols = 0;
        data = nullptr;
    }

    Blob(size_t rows_, size_t cols_)
    {
        rows = rows_;
        cols = cols_;
        data = malloc(rows * cols * sizeof(value_type));
    }


    T& operator() (Index i, Index j)
    {
        return  reinterpret_cast<T*>(data)[i * cols + j];
    }
    const T& operator() (Index i, Index j)const
    {
        return  reinterpret_cast<T*>(data)[i * cols + j];
    }
    void Release() const
    {
        free(data);
    }


    void* data;
};

// 使用using语句， 固定模板类的某些参数为指定的类型
using KeyPointsBlob = Blob<float>;

using KeyPointsColorBlob = Blob<uchar>;

using DescriptorsBlob = Blob<float>;

using MatchesBlob = Blob<point2D_t>;




void SwapMatchesBlob(MatchesBlob& matches)
{
    for(size_t i = 0; i < matches.rows; ++i)
    {
        std::swap(matches(i, 0), matches(i, 1));
    }
}


Database::Image ReadImageRow(sqlite3_stmt* sql_stmt)
{
    Database::Image image;
    image.id = static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0));
    image.name =std::string(reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1)));

    return image;
}




KeyPointsBlob KeyPointsToBlob(const std::vector<cv::KeyPoint>& keypoints)
{
    const typename KeyPointsBlob::Index kNumCols = 4;
    KeyPointsBlob blob(keypoints.size(), kNumCols);
    for(size_t i = 0; i < keypoints.size(); ++i)
    {
        blob(i, 0) = keypoints[i].pt.x;
        blob(i, 1) = keypoints[i].pt.y;
        blob(i, 2) = keypoints[i].size;
        blob(i, 3) = keypoints[i].angle;
    }
    return blob;
}

std::vector<cv::KeyPoint> KeyPointsFromBlob(const KeyPointsBlob& blob)
{
    assert(blob.cols == 4);
    std::vector<cv::KeyPoint> keypoints(blob.rows);
    for(size_t i = 0; i < blob.rows; ++i)
    {
        keypoints[i].pt.x = blob(i, 0);
        keypoints[i].pt.y = blob(i, 1);
        keypoints[i].size = blob(i, 2);
        keypoints[i].angle = blob(i, 3);

    }
    return keypoints;
}

KeyPointsColorBlob KeyPointsColorToBlob(const std::vector<cv::Vec3b>& keypoints_color)
{
    const typename KeyPointsColorBlob::Index kNumCols = 3;
    KeyPointsColorBlob blob(keypoints_color.size(), kNumCols);
    for(size_t i = 0; i < keypoints_color.size(); ++i)
    {
        blob(i, 0) = keypoints_color[i][0];
        blob(i, 1) = keypoints_color[i][1];
        blob(i, 2) = keypoints_color[i][2];

    }
    return blob;
}

std::vector<cv::Vec3b> KeyPointsColorFromBlob(const KeyPointsColorBlob& blob)
{
    assert(blob.cols == 3);
    std::vector<cv::Vec3b> keypoints_color(blob.rows);
    for(size_t i = 0; i < blob.rows; ++i)
    {
        keypoints_color[i][0] = blob(i, 0);
        keypoints_color[i][1] = blob(i, 1);
        keypoints_color[i][2] = blob(i, 2);

    }
    return keypoints_color;
}




DescriptorsBlob DescriptorsToBlob(const cv::Mat& desc)
{
    assert(desc.type() == CV_32F);
    DescriptorsBlob blob(desc.rows, desc.cols);
    for(int i = 0; i < desc.rows; ++i)
    {
        for(int j = 0; j < desc.cols; ++j)
        {
            blob(i, j) = desc.at<float>(i, j);
        }
    }
    return blob;
}

cv::Mat DescriptorsFromBolb(const DescriptorsBlob& blob)
{
    cv::Mat desc(blob.rows, blob.cols, CV_32F);
    for(size_t i = 0; i < blob.rows; ++i)
    {
        for(size_t j = 0; j < blob.cols; ++j)
        {
            desc.at<float>(i, j) = blob(i, j);
        }
    }
    return desc;
}

MatchesBlob MatchesToBlob(const std::vector<cv::DMatch>& matches)
{
    const MatchesBlob::Index kNumCols = 2;
    MatchesBlob blob(matches.size(), kNumCols);
    for(size_t i = 0; i < matches.size(); ++i)
    {
        blob(i, 0) = matches[i].queryIdx;
        blob(i, 1) = matches[i].trainIdx;
    }
    return blob;
}


std::vector<cv::DMatch> MatchesFromBlob(const MatchesBlob& blob)
{
    assert(blob.cols == 2);
    std::vector<cv::DMatch> matches(blob.rows);
    for(size_t i = 0; i < blob.rows; ++i)
    {
        matches[i].queryIdx = blob(i, 0);
        matches[i].trainIdx = blob(i, 1);
    }
    return matches;

}




template <typename BlobType>
BlobType ReadBlob(sqlite3_stmt* sql_stmt, const int rc, const int col)
{
    assert(col >= 0);

    assert(rc == SQLITE_ROW);

    const size_t rows =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 0));
    const size_t cols =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 1));


    assert(rows >= 0);
    assert(cols >= 0);
    BlobType blob(rows, cols);

    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col + 2));

    assert(blob.rows * blob.cols * sizeof(typename BlobType::value_type) == num_bytes);
    memcpy(reinterpret_cast<char*>(blob.data),
           sqlite3_column_blob(sql_stmt, col + 2), num_bytes);


  return blob;
}






template <typename BlobType>
void WriteBlob(sqlite3_stmt* sql_stmt, const BlobType& blob,
                     const int col)
{
  assert(blob.rows >= 0);
  assert(blob.cols >= 0);
  assert(col >= 0);

  const size_t num_bytes = blob.rows * blob.cols * sizeof(typename BlobType::value_type);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 0, blob.rows));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 1, blob.cols));
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt, col + 2,
                                 reinterpret_cast<const char*>(blob.data),
                                 static_cast<int>(num_bytes), SQLITE_STATIC));
}




////////////////////////////////////////////////////////////////////////////////
// 开启/关闭　　　数据库/事务
////////////////////////////////////////////////////////////////////////////////
Database::Database()
{
    database_ = nullptr;
}
Database::~Database()
{
    Close();
}
void Database::Open(const std::string &path)
{
    SQLITE3_CALL(sqlite3_open_v2(path.c_str(), &database_,
                 SQLITE_OPEN_CREATE | SQLITE_OPEN_READWRITE | SQLITE_OPEN_NOMUTEX, nullptr));

    SQLITE3_EXEC(database_, "PRAGMA synchronous=OFF", nullptr);
    SQLITE3_EXEC(database_, "PRAGMA journal_mode=WAL", nullptr);
    SQLITE3_EXEC(database_, "PRAGMA temp_store=MEMORY", nullptr);
    SQLITE3_EXEC(database_, "PRAGMA foreign_keys=ON", nullptr);

    CreateTables();
    UpdateSchema();

    PrepareSQLStatements();
}

void Database::Close()
{
    if(database_ != nullptr)
    {
        FinalizeSQLStatements();
        sqlite3_close_v2(database_);
        database_ = nullptr;
    }
}

void Database::BeginTransaction() const
{
    SQLITE3_EXEC(database_, "BEGIN TRANSACTION", nullptr);
}
void Database::EndTransaction() const
{
    SQLITE3_EXEC(database_, "END TRANSACTION", nullptr);
}






////////////////////////////////////////////////////////////////////////////////
// 读写数据库
////////////////////////////////////////////////////////////////////////////////
bool Database::ExistImageById(const image_t image_id) const
{
    return ExistRowId(sql_stmt_exists_image_id_, image_id);
}
bool Database::ExistImageByName(const std::string name) const
{
    return ExistRowString(sql_stmt_exists_image_name_, name);
}
bool Database::ExistKeyPoints(const image_t image_id) const
{
    return ExistRowId(sql_stmt_exists_keypoints_, image_id);
}
bool Database::ExistKeyPointsColor(const image_t image_id) const
{
    return ExistRowId(sql_stmt_exists_keypoints_color_, image_id);
}
bool Database::ExistDescriptors(const image_t image_id) const
{
    return ExistRowId(sql_stmt_exists_descriptors_, image_id);
}
bool Database::ExistMatches(const image_pair_t pair_id) const
{
    return ExistRowId(sql_stmt_exists_matches_, pair_id);
}
bool Database::ExistMatches(const image_t image_id1, const image_t image_id2) const
{
    image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    return ExistMatches(pair_id);
}



size_t Database::NumImages() const
{
    return CountRows("images");
}
size_t Database::NumKeyPoints(const image_t image_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_num_, 1, image_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_num_));
    assert(rc == SQLITE_ROW);

    size_t num = sqlite3_column_int64(sql_stmt_read_keypoints_num_, 0);

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_num_));

    return num;
}

size_t Database::NumKeyPointsColor(const image_t image_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_color_num_, 1, image_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_color_num_));
    assert(rc == SQLITE_ROW);

    size_t num = sqlite3_column_int64(sql_stmt_read_keypoints_color_num_, 0);

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_color_num_));

    return num;
}

size_t Database::NumDescriptors(const image_t image_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptors_num_, 1, image_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptors_num_));

    assert(rc == SQLITE_ROW);

    size_t num = sqlite3_column_int64(sql_stmt_read_descriptors_num_, 0);

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_descriptors_num_));

    return num;
}
size_t Database::NumMatches(const image_pair_t pair_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_matches_num_, 1, pair_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_num_));

    assert(rc == SQLITE_ROW);

    size_t num = sqlite3_column_int64(sql_stmt_read_matches_num_, 0);

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_num_));

    return num;

}
size_t Database::NumMatches(const image_t image_id1, const image_t image_id2) const
{
    image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    return NumMatches(pair_id);
}


Database::Image Database::ReadImageById(const image_t image_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_image_id_, 1, image_id));

    Database::Image image;


    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_id_));
    if(rc == SQLITE_ROW)
    {
        image = ReadImageRow(sql_stmt_read_image_id_);
    }

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_id_));
    return image;
}
Database::Image Database::ReadImageByName(const std::string name) const
{
    SQLITE3_CALL(sqlite3_bind_text(sql_stmt_read_image_name_, 1, name.c_str(),
                                   static_cast<int>(name.size()), SQLITE_STATIC));

    Database::Image image;

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_name_));
    if(rc == SQLITE_ROW)
    {
        image = ReadImageRow(sql_stmt_read_image_name_);
    }
    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_name_));

    return image;
}
std::vector<Database::Image> Database::ReadAllImages() const
{
    std::vector<Database::Image> images;
    images.reserve(NumImages());

    while(SQLITE3_CALL(sqlite3_step(sql_stmt_read_images_)) == SQLITE_ROW)
    {
        images.push_back(ReadImageRow(sql_stmt_read_images_));
    }
    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_images_));

    return images;
}
std::vector<cv::KeyPoint> Database::ReadKeyPoints(const image_t image_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));
    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));

    const KeyPointsBlob blob =
            ReadBlob<KeyPointsBlob>(sql_stmt_read_keypoints_, rc, 0);
    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_));

    const std::vector<cv::KeyPoint>&& keypoints = KeyPointsFromBlob(blob);
    blob.Release();
    return keypoints;
}

std::vector<cv::Vec3b> Database::ReadKeyPointsColor(const image_t image_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_color_, 1, image_id));
    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_color_));

    const KeyPointsColorBlob blob =
            ReadBlob<KeyPointsColorBlob>(sql_stmt_read_keypoints_color_, rc, 0);
    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_color_));

    const std::vector<cv::Vec3b>&& keypoints_color = KeyPointsColorFromBlob(blob);
    blob.Release();
    return keypoints_color;
}

cv::Mat Database::ReadDescriptors(const image_t image_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptors_, 1, image_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptors_));
    const DescriptorsBlob blob =
            ReadBlob<DescriptorsBlob>(sql_stmt_read_descriptors_, rc, 0);

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_descriptors_));

    const cv::Mat&& desc = DescriptorsFromBolb(blob);
    blob.Release();
    return desc;
}
std::vector<cv::DMatch> Database::ReadMatches(const image_pair_t pair_id) const
{
    image_t image_id1, image_id2;
    PairIdToImagePair(pair_id, &image_id1, &image_id2);
    return std::move(ReadMatches(image_id1, image_id2));
}
std::vector<cv::DMatch> Database::ReadMatches(const image_t image_id1, const image_t image_id2) const
{
    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_matches_, 1, pair_id));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_));
    MatchesBlob blob =
            ReadBlob<MatchesBlob>(sql_stmt_read_matches_, rc, 0);

    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_));

    if(SwapImagePair(image_id1, image_id2))
    {
        SwapMatchesBlob(blob);
    }
    const std::vector<cv::DMatch>&& matches = MatchesFromBlob(blob);
    blob.Release();
    return matches;
}

std::vector<std::pair<image_pair_t, std::vector<cv::DMatch>>> Database::ReadAllMatches() const
{
    std::vector<std::pair<image_pair_t, std::vector<cv::DMatch>>> results;

    int rc;
    while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_all_))) ==  SQLITE_ROW)
    {
        const image_pair_t pair_id = static_cast<image_pair_t>(sqlite3_column_int64(sql_stmt_read_matches_all_, 0));
        const MatchesBlob blob =
          ReadBlob<MatchesBlob>(sql_stmt_read_matches_all_, rc, 1);
        results.emplace_back(pair_id, MatchesFromBlob(blob));
    }


    SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_all_));

    return results;

}


image_t Database::WriteImage(const Database::Image& image, const bool use_image_id) const
{
    if(use_image_id)
    {
        assert(!ExistImageById(image.id));
        SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_image_, 1, image.id));
    }
    else
    {
        SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_image_, 1));
    }


    SQLITE3_CALL(sqlite3_bind_text(sql_stmt_add_image_, 2, image.name.c_str(),
                                   static_cast<int>(image.name.size()), SQLITE_STATIC));
    SQLITE3_CALL(sqlite3_step(sql_stmt_add_image_));
    SQLITE3_CALL(sqlite3_reset(sql_stmt_add_image_));

    return static_cast<image_t>(sqlite3_last_insert_rowid(database_));
}
void Database::WriteKeyPoints(const image_t image_id, const std::vector<cv::KeyPoint>& keypoints) const
{
    const KeyPointsBlob blob = KeyPointsToBlob(keypoints);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_keypoints_, 1, image_id));
    WriteBlob(sql_stmt_add_keypoints_, blob, 2);

    SQLITE3_CALL(sqlite3_step(sql_stmt_add_keypoints_));
    SQLITE3_CALL(sqlite3_reset(sql_stmt_add_keypoints_));

    blob.Release();
}

void Database::WriteKeyPointsColor(const image_t image_id, const std::vector<cv::Vec3b>& keypoints_color) const
{
    const KeyPointsColorBlob blob = KeyPointsColorToBlob(keypoints_color);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_keypoints_color_, 1, image_id));
    WriteBlob(sql_stmt_add_keypoints_color_, blob, 2);

    SQLITE3_CALL(sqlite3_step(sql_stmt_add_keypoints_color_));
    SQLITE3_CALL(sqlite3_reset(sql_stmt_add_keypoints_color_));

    blob.Release();
}


void Database::WriteDescriptors(const image_t image_id, const cv::Mat& descriptors) const
{
    const DescriptorsBlob blob = DescriptorsToBlob(descriptors);

    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_descriptors_, 1, image_id));
    WriteBlob(sql_stmt_add_descriptors_, blob, 2);

    SQLITE3_CALL(sqlite3_step(sql_stmt_add_descriptors_));
    SQLITE3_CALL(sqlite3_reset(sql_stmt_add_descriptors_));

    blob.Release();
}

void Database::WriteMatches(const image_t image_id1, const image_t image_id2, const std::vector<cv::DMatch>& matches) const
{
    const image_pair_t pair_id = ImagePairToPairId(image_id1, image_id2);
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_matches_, 1, pair_id));

    MatchesBlob blob = MatchesToBlob(matches);
    if(SwapImagePair(image_id1, image_id2))
    {
        SwapMatchesBlob(blob);
    }

    WriteBlob(sql_stmt_add_matches_, blob, 2);

    SQLITE3_CALL(sqlite3_step(sql_stmt_add_matches_));
    SQLITE3_CALL(sqlite3_reset(sql_stmt_add_matches_));

    blob.Release();
}




////////////////////////////////////////////////////////////////////////////////
// image_t  <---> image_pair_t
////////////////////////////////////////////////////////////////////////////////
image_pair_t Database::ImagePairToPairId(const image_t image_id1,
                                         const image_t image_id2)
{

    assert(image_id1 >= 0);
    assert(image_id2 >= 0);
    assert(image_id1 < kMaxNumImages);
    assert(image_id2 < kMaxNumImages);

    // image_id1 > image_id2
    if(SwapImagePair(image_id1, image_id2))
    {
        return kMaxNumImages * image_id2 + image_id1;
    }
    // image_id2 > image_id1
    else
    {
        return kMaxNumImages * image_id1 + image_id2;
    }
}

void Database::PairIdToImagePair(const image_pair_t pair_id,
                                 image_t* image_id1,
                                 image_t* image_id2)
{
    // default image_id1 < image_2
    *image_id2 = static_cast<image_t>(pair_id % kMaxNumImages);
    *image_id1 = static_cast<image_t>(pair_id - *image_id2) / kMaxNumImages;

    assert(*image_id1 >= 0);
    assert(*image_id2 >= 0);
    assert(*image_id1 < kMaxNumImages);
    assert(*image_id2 < kMaxNumImages);
}

bool Database::SwapImagePair(const image_t image_id1,
                             const image_t image_id2)
{
    return image_id1 > image_id2;
}



////////////////////////////////////////////////////////////////////////////////
// 创建数据库表格
////////////////////////////////////////////////////////////////////////////////
void Database::CreateTables() const
{
    CreateImageTable();
    CreateKeyPointsTable();
    CreateKeyPointsColorTable();
    CreateDescriptorsTable();
    CreateMatchesTable();
}

void Database::CreateImageTable() const
{
   const std::string sql =
           "CREATE TABLE IF NOT EXISTS images"
           "(  image_id  INTEGER PRIMARY KEY AUTOINCREMENT   NOT NULL,"
           "   name      TEXT                                NOT NULL UNIQUE)";
   SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateKeyPointsTable() const
{
    const std::string sql =
            "CREATE TABLE IF NOT EXISTS keypoints"
            "  (image_id    INTEGER    PRIMARY KEY    NOT NULL,"
            "   rows        INTEGER                   NOT NULL,"
            "   cols        INTEGER                   NOT NULL,"
            "   data        BLOB,"
            "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)";
    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateKeyPointsColorTable() const
{
    const std::string sql =
            "CREATE TABLE IF NOT EXISTS colors"
            "  (image_id    INTEGER    PRIMARY KEY    NOT NULL,"
            "   rows        INTEGER                   NOT NULL,"
            "   cols        INTEGER                   NOT NULL,"
            "   data        BLOB,"
            "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)";
    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateDescriptorsTable() const
{
    const std::string sql  =
            "CREATE TABLE IF NOT EXISTS descriptors"
            "   (image_id   INTEGER    PRIMARY KEY    NOT NULL,"
            "    rows       INTEGER                   NOT NULL,"
            "    cols       INTEGER                   NOT NULL,"
            "    data       BLOB,"
            "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)";
    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void Database::CreateMatchesTable() const
{
    const std::string sql =
            "CREATE TABLE IF NOT EXISTS matches"
            "   (pair_id    INTEGER    PRIMARY KEY    NOT NULL,"
            "    rows       INTEGER                   NOT NULL,"
            "    cols       INTEGER                   NOT NULL,"
            "    data       BLOB);";
    SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}


////////////////////////////////////////////////////////////////////////////////
// 准备/销毁　数据库语句
////////////////////////////////////////////////////////////////////////////////
void Database::PrepareSQLStatements()
{
    sql_stmts_.clear();

    std::string sql;

    //////////////////////////////////////////////////////////////////////////////
    // exists_*
    //////////////////////////////////////////////////////////////////////////////
    sql = "SELECT 1 FROM images WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_exists_image_id_, 0));
    sql_stmts_.push_back(sql_stmt_exists_image_id_);

    sql = "SELECT 1 FROM images WHERE name = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_exists_image_name_, 0));
    sql_stmts_.push_back(sql_stmt_exists_image_name_);

    sql = "SELECT 1 FROM keypoints WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_exists_keypoints_, 0));
    sql_stmts_.push_back(sql_stmt_exists_keypoints_);


    sql = "SELECT 1 FROM colors WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_exists_keypoints_color_, 0));
    sql_stmts_.push_back(sql_stmt_exists_keypoints_color_);

    sql = "SELECT 1 FROM descriptors WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_exists_descriptors_, 0));
    sql_stmts_.push_back(sql_stmt_exists_descriptors_);

    sql = "SELECT 1 FROM matches WHERE pair_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_exists_matches_, 0));
    sql_stmts_.push_back(sql_stmt_exists_matches_);




    //////////////////////////////////////////////////////////////////////////////
    // read_*
    //////////////////////////////////////////////////////////////////////////////
    sql = "SELECT * FROM images WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_image_id_, 0));
    sql_stmts_.push_back(sql_stmt_read_image_id_);

    sql = "SELECT * FROM images WHERE name = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_image_name_, 0));
    sql_stmts_.push_back(sql_stmt_read_image_name_);

    sql = "SELECT * FROM images;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_images_, 0));
    sql_stmts_.push_back(sql_stmt_read_images_);

    sql = "SELECT rows, cols, data FROM keypoints WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_keypoints_, 0));
    sql_stmts_.push_back(sql_stmt_read_keypoints_);


    sql = "SELECT rows FROM keypoints WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_keypoints_num_, 0));
    sql_stmts_.push_back(sql_stmt_read_keypoints_num_);



    sql = "SELECT rows, cols, data FROM colors WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_keypoints_color_, 0));
    sql_stmts_.push_back(sql_stmt_read_keypoints_color_);



    sql = "SELECT rows FROM colors WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_keypoints_color_num_, 0));
    sql_stmts_.push_back(sql_stmt_read_keypoints_color_num_);



    sql = "SELECT rows, cols, data FROM descriptors WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_descriptors_, 0));
    sql_stmts_.push_back(sql_stmt_read_descriptors_);

    sql = "SELECT rows FROM descriptors WHERE image_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_descriptors_num_, 0));
    sql_stmts_.push_back(sql_stmt_read_descriptors_num_);


    sql = "SELECT rows, cols, data FROM matches WHERE pair_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_matches_, 0));
    sql_stmts_.push_back(sql_stmt_read_matches_);

    sql = "SELECT rows FROM matches WHERE pair_id = ?;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_matches_num_, 0));
    sql_stmts_.push_back(sql_stmt_read_matches_num_);

    sql = "SELECT * FROM matches WHERE rows > 0;";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_read_matches_all_, 0));
    sql_stmts_.push_back(sql_stmt_read_matches_all_);





    //////////////////////////////////////////////////////////////////////////////
    // add_*
    //////////////////////////////////////////////////////////////////////////////
    sql = "INSERT INTO images(image_id, name) VALUES(?, ?);";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_image_, 0));
    sql_stmts_.push_back(sql_stmt_add_image_);

    sql = "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_keypoints_, 0));
    sql_stmts_.push_back(sql_stmt_add_keypoints_);


    sql = "INSERT INTO colors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_keypoints_color_, 0));
    sql_stmts_.push_back(sql_stmt_add_keypoints_color_);

    sql = "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_descriptors_, 0));
    sql_stmts_.push_back(sql_stmt_add_descriptors_);

    sql = "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);";
    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_matches_, 0));
    sql_stmts_.push_back(sql_stmt_add_matches_);


}

void Database::FinalizeSQLStatements()
{
    for(sqlite3_stmt* sql_stmt : sql_stmts_)
    {
        SQLITE3_CALL(sqlite3_finalize(sql_stmt));
    }
}


void Database::UpdateSchema() const
{
    // Query user_version
    const std::string query_user_version_sql = "PRAGMA user_version;";
    sqlite3_stmt* query_user_version_sql_stmt;
    SQLITE3_CALL(sqlite3_prepare_v2(database_, query_user_version_sql.c_str(), -1,
                                  &query_user_version_sql_stmt, 0));

    // Update schema, if user_version < kSchemaVersion
    if (SQLITE3_CALL(sqlite3_step(query_user_version_sql_stmt)) == SQLITE_ROW) {
        const int user_version = sqlite3_column_int(query_user_version_sql_stmt, 0);
        // user_version == 0: initial value from SQLite, nothing to do, since all
        // tables were created in `Database::CreateTables`
        if (user_version > 0) {
          // if (user_version < 2) {}
        }
    }

    SQLITE3_CALL(sqlite3_finalize(query_user_version_sql_stmt));

    const std::string update_user_version_sql =
      "PRAGMA user_version = " + std::to_string(kSchemaVersion) + ";";
    SQLITE3_EXEC(database_, update_user_version_sql.c_str(), nullptr);
}


bool Database::ExistRowId(sqlite3_stmt* sql_stmt, const size_t row_id) const
{
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, 1, static_cast<sqlite3_int64>(row_id)));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));

    const bool exists = (rc == SQLITE_ROW);

    SQLITE3_CALL(sqlite3_reset(sql_stmt));

    return exists;
}
bool Database::ExistRowString(sqlite3_stmt* sql_stmt, const std::string& row_entry) const
{
    SQLITE3_CALL(sqlite3_bind_text(sql_stmt, 1, row_entry.c_str(),
                                    static_cast<int>(row_entry.size()), SQLITE_STATIC));

    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));

    const bool exists = (rc == SQLITE_ROW);

    SQLITE3_CALL(sqlite3_reset(sql_stmt));

    return exists;
}

size_t Database::CountRows(const std::string& table) const
{
    const std::string sql = "SELECT COUNT(*) FROM " + table + ";";
    sqlite3_stmt* sql_stmt;

    SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

    size_t count = 0;
    const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
    if(rc == SQLITE_ROW)
    {
        count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
    }

    SQLITE3_CALL(sqlite3_finalize(sql_stmt));

    return count;
}
