#include <iostream>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <hdl_global_localization/srv/set_global_map.hpp>
#include <hdl_global_localization/srv/set_global_localization_engine.hpp>
#include <hdl_global_localization/srv/query_global_localization.hpp>

#include <hdl_global_localization/util/config.hpp>
#include <hdl_global_localization/engines/global_localization_bbs.hpp>
#include <hdl_global_localization/engines/global_localization_fpfh_ransac.hpp>

namespace hdl_global_localization {

class GlobalLocalizationNode : public rclcpp::Node {
public:
  GlobalLocalizationNode(rclcpp::NodeOptions& options) : rclcpp::Node("hdl_global_localization_node", options) {
    engine_name_used = this->declare_parameter<std::string>("engine_name_used", "BBS");

    // BBS Parameters
    // const std::string config_path = ament_index_cpp::get_package_share_directory("hdl_global_localization") + "/config";
    const std::string config_path = this->declare_parameter<std::string>("config_path", ament_index_cpp::get_package_share_directory("hdl_global_localization") + "/config");

    // FPFH_RANSAC Parameters
    normal_estimation_radius = this->declare_parameter<double>("fpfh/normal_estimation_radius", 2.0);
    search_radius = this->declare_parameter<double>("fpfh/search_radius", 8.0);
    voxel_based = this->declare_parameter<bool>("ransac/voxel_based", true);
    max_correspondence_distance = this->declare_parameter<double>("ransac/max_correspondence_distance", 1.0);
    similarity_threshold = this->declare_parameter<double>("ransac/similarity_threshold", 0.5);
    correspondence_randomness = this->declare_parameter<int>("ransac/correspondence_randomness", 2);
    max_iterations = this->declare_parameter<int>("ransac/max_iterations", 100000);
    matching_budget = this->declare_parameter<int>("ransac/matching_budget", 10000);
    min_inlier_fraction = this->declare_parameter<double>("ransac/min_inlier_fraction", 0.25);

    GlobalConfig::instance(config_path);

    const Config config(GlobalConfig::get_config_path("config_base"));
    globalmap_downsample_resolution = config.param<double>("base", "globalmap_downsample_resolution", 0.5);
    query_downsample_resolution = config.param<double>("base", "query_downsample_resolution", 0.5);

    if (engine_name_used == "BBS"){
      set_engine("BBS");
    } else if (engine_name_used == "FPFH_RANSAC") {
      set_engine("FPFH_RANSAC");
    } else {
      set_engine("BBS");
    }

    using std::placeholders::_1;
    using std::placeholders::_2;
    set_engine_service = this->create_service<srv::SetGlobalLocalizationEngine>("set_engine", std::bind(&GlobalLocalizationNode::set_global_localization_engine, this, _1, _2));
    set_global_map_service = this->create_service<srv::SetGlobalMap>("set_global_map", std::bind(&GlobalLocalizationNode::set_global_map, this, _1, _2));
    query_service = this->create_service<srv::QueryGlobalLocalization>("query", std::bind(&GlobalLocalizationNode::query, this, _1, _2));
  }

  void set_global_localization_engine(const srv::SetGlobalLocalizationEngine::Request::SharedPtr req, srv::SetGlobalLocalizationEngine::Response::SharedPtr res) {
    set_engine(req->engine_name.data);
  }

  bool set_engine(const std::string& engine_name) {
    RCLCPP_INFO_STREAM(this->get_logger(), "Set Global Localization Engine (" << engine_name << ")");
    if (engine_name == "BBS") {
      engine.reset(new GlobalLocalizationBBS(*this));
    } else if (engine_name == "FPFH_RANSAC") {
      // Parameter
      GlobalLocalizationEngineFPFH_RANSACParams params;
      params.normal_estimation_radius = normal_estimation_radius;
      params.search_radius = search_radius;

      params.ransac_params.voxel_based = voxel_based;
      params.ransac_params.max_correspondence_distance = max_correspondence_distance;
      params.ransac_params.similarity_threshold = similarity_threshold;
      params.ransac_params.correspondence_randomness = correspondence_randomness;
      params.ransac_params.max_iterations = max_iterations;
      params.ransac_params.matching_budget = matching_budget;
      params.ransac_params.min_inlier_fraction = min_inlier_fraction;

      engine.reset(new GlobalLocalizationEngineFPFH_RANSAC(params));
    } else {
      RCLCPP_WARN_STREAM(this->get_logger(), "Unknown Global Localization Engine:" << engine_name);
      return false;
    }

    if (global_map) {
      engine->set_global_map(global_map);
    }

    return true;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double resolution) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(resolution, resolution, resolution);
    voxelgrid.setInputCloud(cloud);
    voxelgrid.filter(*filtered);
    return filtered;
  }

  void set_global_map(const srv::SetGlobalMap::Request::SharedPtr req, srv::SetGlobalMap::Response::SharedPtr res) {
    RCLCPP_INFO_STREAM(this->get_logger(), "Global Map Received");

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(req->global_map, *cloud);
    cloud = downsample(cloud, globalmap_downsample_resolution);

    globalmap_header = req->global_map.header;
    global_map = cloud;
    engine->set_global_map(global_map);

    RCLCPP_INFO_STREAM(this->get_logger(), "DONE");
  }

  void query(const srv::QueryGlobalLocalization::Request::SharedPtr req, srv::QueryGlobalLocalization::Response::SharedPtr res) {
    RCLCPP_INFO_STREAM(this->get_logger(), "Query Global Localization");
    if (global_map == nullptr) {
      RCLCPP_WARN_STREAM(this->get_logger(), "No Globalmap");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(req->cloud, *cloud);
    cloud = downsample(cloud, query_downsample_resolution);

    auto results = engine->query(cloud, req->max_num_candidates);

    res->inlier_fractions.resize(results.results.size());
    res->errors.resize(results.results.size());
    res->poses.resize(results.results.size());

    res->header = req->cloud.header;
    res->globalmap_header = globalmap_header;

    for (int i = 0; i < results.results.size(); i++) {
      const auto& result = results.results[i];
      Eigen::Quaternionf quat(result->pose.linear());
      Eigen::Vector3f trans(result->pose.translation());

      res->inlier_fractions[i] = result->inlier_fraction;
      res->errors[i] = result->error;
      res->poses[i].orientation.x = quat.x();
      res->poses[i].orientation.y = quat.y();
      res->poses[i].orientation.z = quat.z();
      res->poses[i].orientation.w = quat.w();

      res->poses[i].position.x = trans.x();
      res->poses[i].position.y = trans.y();
      res->poses[i].position.z = trans.z();
    }
  }

private:
  double globalmap_downsample_resolution;
  double query_downsample_resolution;

  double normal_estimation_radius;
  double search_radius;

  bool voxel_based;
  double max_correspondence_distance;
  double similarity_threshold;
  int correspondence_randomness;
  int max_iterations;
  int matching_budget;
  double min_inlier_fraction;
  std::string engine_name_used;

  rclcpp::ServiceBase::SharedPtr set_engine_service;
  rclcpp::ServiceBase::SharedPtr set_global_map_service;
  rclcpp::ServiceBase::SharedPtr query_service;

  std_msgs::msg::Header globalmap_header;
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_map;
  std::unique_ptr<GlobalLocalizationEngine> engine;
};

}  // namespace hdl_global_localization

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  auto node = std::make_shared<hdl_global_localization::GlobalLocalizationNode>(options);
  rclcpp::spin(node);

  return 0;
}