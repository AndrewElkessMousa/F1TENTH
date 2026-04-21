#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "frenet_tenth_planner/frenet_optimal_trajectory.hpp"
#include "frenet_tenth_planner/track_loader.hpp"

using std::placeholders::_1;
namespace ftp = frenet_tenth_planner;

class FrenetPlannerNode : public rclcpp::Node
{
public:
  FrenetPlannerNode()
  : Node("frenet_planner_node")
  {
    declare_parameter<std::string>("odom_topic", "/ego_racecar/odom");
    declare_parameter<std::string>("path_topic", "/frenet_path");
    declare_parameter<std::string>("target_topic", "/frenet_target_point");
    declare_parameter<std::string>("markers_topic", "/frenet_debug_markers");
    declare_parameter<std::string>("centerline_topic", "/centerline_path");
    declare_parameter<std::string>("frame_id", "map");
    declare_parameter<std::string>("centerline_csv", "");
    declare_parameter<double>("track_width", 1.8);
    declare_parameter<double>("target_speed", 3.0);
    declare_parameter<double>("lookahead_index", 6.0);
    declare_parameter<double>("max_speed", 6.0);
    declare_parameter<double>("max_accel", 6.0);
    declare_parameter<double>("max_curvature", 5.0);
    declare_parameter<double>("max_road_width", 0.8);
    declare_parameter<double>("d_road_w", 0.1);
    declare_parameter<double>("dt", 0.1);
    declare_parameter<double>("maxt", 2.0);
    declare_parameter<double>("mint", 1.0);
    declare_parameter<double>("d_t_s", 0.5);
    declare_parameter<double>("n_s_sample", 1.0);
    declare_parameter<double>("robot_radius", 0.25);
    declare_parameter<double>("safe_distance", 0.05);
    declare_parameter<double>("range_path_check", 1.0);
    declare_parameter<double>("next_s_borders", 30.0);
    declare_parameter<double>("kj", 0.01);
    declare_parameter<double>("kt", 0.1);
    declare_parameter<double>("kd", 1.5);
    declare_parameter<double>("klat", 1.0);
    declare_parameter<double>("klon", 1.0);
    declare_parameter<bool>("check_derivatives", true);
    declare_parameter<int>("overtake_strategy", 0);
    declare_parameter<bool>("debug_log", true);

    // Fake obstacle test parameters
    declare_parameter<bool>("use_test_obstacle", true);
    declare_parameter<double>("test_obstacle_s_offset", 2.0);
    declare_parameter<double>("test_obstacle_d_offset", 0.0);
    declare_parameter<double>("test_obstacle_radius", 0.20);
    declare_parameter<double>("debug_min_speed", 2.0);
    declare_parameter<bool>("force_debug_speed_when_stopped", true);

    const auto centerline_csv = get_parameter("centerline_csv").as_string();
    const auto track_width = get_parameter("track_width").as_double();

    if (centerline_csv.empty()) {
      throw std::runtime_error("Parameter 'centerline_csv' must point to your centerline CSV file.");
    }

    track_ = ftp::load_track_from_csv(centerline_csv, track_width, true);

    if (track_.center_x.empty() || track_.center_y.empty()) {
      throw std::runtime_error("Loaded centerline is empty. Check your CSV file.");
    }

    frame_id_ = get_parameter("frame_id").as_string();
    lookahead_index_ = static_cast<size_t>(get_parameter("lookahead_index").as_double());
    overtake_strategy_ = get_parameter("overtake_strategy").as_int();
    debug_log_ = get_parameter("debug_log").as_bool();

    use_test_obstacle_ = get_parameter("use_test_obstacle").as_bool();
    test_obstacle_s_offset_ = get_parameter("test_obstacle_s_offset").as_double();
    test_obstacle_d_offset_ = get_parameter("test_obstacle_d_offset").as_double();
    test_obstacle_radius_ = get_parameter("test_obstacle_radius").as_double();
    debug_min_speed_ = get_parameter("debug_min_speed").as_double();
    force_debug_speed_when_stopped_ = get_parameter("force_debug_speed_when_stopped").as_bool();

    ftp::vecD tx, ty, tyaw, tc;
    spline_ref_ = ftp::calc_spline_course(track_.center_x, track_.center_y, tx, ty, tyaw, tc, 0.1);

    tx.clear(); ty.clear(); tyaw.clear(); tc.clear();
    spline_left_ = ftp::calc_spline_course(track_.left_x, track_.left_y, tx, ty, tyaw, tc, 0.1);

    tx.clear(); ty.clear(); tyaw.clear(); tc.clear();
    spline_right_ = ftp::calc_spline_course(track_.right_x, track_.right_y, tx, ty, tyaw, tc, 0.1);

    ftp::Params<double> params{};
    params.max_speed = get_parameter("max_speed").as_double();
    params.max_accel = get_parameter("max_accel").as_double();
    params.max_curvature = get_parameter("max_curvature").as_double();
    params.max_road_width = get_parameter("max_road_width").as_double();
    params.d_road_w = get_parameter("d_road_w").as_double();
    params.dt = get_parameter("dt").as_double();
    params.maxt = get_parameter("maxt").as_double();
    params.mint = get_parameter("mint").as_double();
    params.target_speed = get_parameter("target_speed").as_double();
    params.d_t_s = get_parameter("d_t_s").as_double();
    params.n_s_sample = get_parameter("n_s_sample").as_double();
    params.robot_radius = get_parameter("robot_radius").as_double();
    params.max_road_width_left = get_parameter("max_road_width").as_double();
    params.max_road_width_right = get_parameter("max_road_width").as_double();
    params.safe_distance = get_parameter("safe_distance").as_double();
    params.range_path_check = get_parameter("range_path_check").as_double();
    params.next_s_borders = get_parameter("next_s_borders").as_double();
    params.kj = get_parameter("kj").as_double();
    params.kt = get_parameter("kt").as_double();
    params.kd = get_parameter("kd").as_double();
    params.klat = get_parameter("klat").as_double();
    params.klon = get_parameter("klon").as_double();
    params.check_derivatives = get_parameter("check_derivatives").as_bool();

    planner_ = std::make_unique<ftp::FrenetPlanner<double>>(params, spline_ref_, spline_left_, spline_right_);

    const auto odom_topic = get_parameter("odom_topic").as_string();
    const auto path_topic = get_parameter("path_topic").as_string();
    const auto target_topic = get_parameter("target_topic").as_string();
    const auto markers_topic = get_parameter("markers_topic").as_string();
    const auto centerline_topic = get_parameter("centerline_topic").as_string();

    path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic, 1);
    target_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(target_topic, 1);

    rclcpp::QoS latched_qos(rclcpp::KeepLast(1));
    latched_qos.transient_local();
    latched_qos.reliable();

    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(markers_topic, latched_qos);
    centerline_pub_ = create_publisher<nav_msgs::msg::Path>(centerline_topic, latched_qos);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, 10, std::bind(&FrenetPlannerNode::odomCallback, this, _1));

    publishTrackMarkers();
    publishCenterlinePath();

    centerline_timer_ = create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&FrenetPlannerNode::publishStaticVisuals, this));

    RCLCPP_INFO(
      get_logger(),
      "Frenet planner ready. centerline points=%zu csv=%s frame=%s",
      track_.center_x.size(), centerline_csv.c_str(), frame_id_.c_str());
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    const double x = msg->pose.pose.position.x;
    const double y = msg->pose.pose.position.y;
    const double vx = msg->twist.twist.linear.x;
    const double vy = msg->twist.twist.linear.y;

    double speed = std::sqrt(vx * vx + vy * vy);
    if (force_debug_speed_when_stopped_ && speed < 0.3) {
      speed = debug_min_speed_;
    }

    double s0 = last_s_guess_;
    double d0 = 0.0;

    spline_ref_.calc_projection(&s0, &d0, x, y, last_s_guess_);

    if (std::abs(s0 - last_s_guess_) > 5.0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Large s jump detected. old=%.3f new=%.3f", last_s_guess_, s0);
    }

    last_s_guess_ = s0;

    std::vector<ftp::Obstacle> obstacles;
    double obs_x = 0.0;
    double obs_y = 0.0;
    bool published_test_obstacle = false;

    if (use_test_obstacle_) {
      double obs_s = test_obstacle_s_offset_;   // fixed position on track, NOT relative to car

      const double s_last = spline_ref_.get_s_last();
      while (obs_s > s_last) {
        obs_s -= s_last;
      }

      double center_x = 0.0;
      double center_y = 0.0;
      spline_ref_.calc_position(&center_x, &center_y, obs_s);

      const double yaw = spline_ref_.calc_yaw(obs_s);

      obs_x = center_x - test_obstacle_d_offset_ * std::sin(yaw);
      obs_y = center_y + test_obstacle_d_offset_ * std::cos(yaw);

      obstacles.push_back(
        ftp::Obstacle{obs_x, obs_y, test_obstacle_radius_, obs_s, test_obstacle_d_offset_});

      published_test_obstacle = true;
    }

    ftp::FrenetPath<double> first, last;
    auto path = planner_->frenet_optimal_planning(
      s0, speed, d0, 0.0, 0.0, obstacles, first, last, overtake_strategy_);

    if (published_test_obstacle) {
      publishObstacleMarkers(obs_x, obs_y, test_obstacle_radius_);
    } else {
      clearObstacleMarkers();
    }

    if (path.empty || path.x.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Planner returned no valid path.");
      return;
    }

    if (debug_log_) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "pose=(%.3f, %.3f) speed=%.3f s=%.3f d=%.3f path_pts=%zu obstacle_count=%zu",
        x, y, speed, s0, d0, path.x.size(), obstacles.size());
    }

    publishPath(path, msg->header.stamp);
    publishTarget(path, msg->header.stamp);
  }

  void publishStaticVisuals()
  {
    publishTrackMarkers();
    publishCenterlinePath();
  }

  void publishPath(const ftp::FrenetPath<double> &path, const rclcpp::Time &stamp)
  {
    nav_msgs::msg::Path msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id_;

    for (size_t i = 0; i < path.x.size(); ++i) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = msg.header;
      pose.pose.position.x = path.x[i];
      pose.pose.position.y = path.y[i];
      pose.pose.position.z = 0.0;

      tf2::Quaternion q;
      const double yaw = (i < path.yaw.size()) ? path.yaw[i] : 0.0;
      q.setRPY(0.0, 0.0, yaw);

      pose.pose.orientation.x = q.x();
      pose.pose.orientation.y = q.y();
      pose.pose.orientation.z = q.z();
      pose.pose.orientation.w = q.w();

      msg.poses.push_back(pose);
    }

    path_pub_->publish(msg);
  }

  void publishTarget(const ftp::FrenetPath<double> &path, const rclcpp::Time &stamp)
  {
    const size_t idx = std::min(lookahead_index_, path.x.size() - 1);

    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id_;
    msg.point.x = path.x[idx];
    msg.point.y = path.y[idx];
    msg.point.z = 0.0;

    target_pub_->publish(msg);
  }

  void publishCenterlinePath()
  {
    nav_msgs::msg::Path msg;
    msg.header.stamp = now();
    msg.header.frame_id = frame_id_;

    for (size_t i = 0; i < track_.center_x.size(); ++i) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = msg.header;
      pose.pose.position.x = track_.center_x[i];
      pose.pose.position.y = track_.center_y[i];
      pose.pose.position.z = 0.0;
      pose.pose.orientation.w = 1.0;
      msg.poses.push_back(pose);
    }

    if (!track_.center_x.empty()) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = msg.header;
      pose.pose.position.x = track_.center_x.front();
      pose.pose.position.y = track_.center_y.front();
      pose.pose.position.z = 0.0;
      pose.pose.orientation.w = 1.0;
      msg.poses.push_back(pose);
    }

    centerline_pub_->publish(msg);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "Published /centerline_path with %zu poses in frame '%s'",
      msg.poses.size(), frame_id_.c_str());
  }

  void publishTrackMarkers()
  {
    visualization_msgs::msg::MarkerArray array;
    array.markers.push_back(makeLineMarker(0, "track", track_.center_x, track_.center_y, 0.05, 0.1, 0.8, 0.1));
    array.markers.push_back(makeLineMarker(1, "track", track_.left_x, track_.left_y, 0.03, 0.8, 0.1, 0.1));
    array.markers.push_back(makeLineMarker(2, "track", track_.right_x, track_.right_y, 0.03, 0.1, 0.1, 0.8));
    markers_pub_->publish(array);
  }

  void publishObstacleMarkers(double x, double y, double radius)
  {
    visualization_msgs::msg::MarkerArray array;

    // Keep track lines visible too
    array.markers.push_back(makeLineMarker(0, "track", track_.center_x, track_.center_y, 0.05, 0.1, 0.8, 0.1));
    array.markers.push_back(makeLineMarker(1, "track", track_.left_x, track_.left_y, 0.03, 0.8, 0.1, 0.1));
    array.markers.push_back(makeLineMarker(2, "track", track_.right_x, track_.right_y, 0.03, 0.1, 0.1, 0.8));

    visualization_msgs::msg::Marker obstacle;
    obstacle.header.frame_id = frame_id_;
    obstacle.header.stamp = now();
    obstacle.ns = "obstacles";
    obstacle.id = 100;
    obstacle.type = visualization_msgs::msg::Marker::SPHERE;
    obstacle.action = visualization_msgs::msg::Marker::ADD;
    obstacle.pose.position.x = x;
    obstacle.pose.position.y = y;
    obstacle.pose.position.z = 0.0;
    obstacle.pose.orientation.w = 1.0;
    obstacle.scale.x = 2.0 * radius;
    obstacle.scale.y = 2.0 * radius;
    obstacle.scale.z = 0.25;
    obstacle.color.a = 1.0;
    obstacle.color.r = 1.0;
    obstacle.color.g = 0.0;
    obstacle.color.b = 0.0;
    array.markers.push_back(obstacle);

    markers_pub_->publish(array);
  }

  void clearObstacleMarkers()
  {
    visualization_msgs::msg::MarkerArray array;

    visualization_msgs::msg::Marker obstacle;
    obstacle.header.frame_id = frame_id_;
    obstacle.header.stamp = now();
    obstacle.ns = "obstacles";
    obstacle.id = 100;
    obstacle.action = visualization_msgs::msg::Marker::DELETE;
    array.markers.push_back(obstacle);

    markers_pub_->publish(array);
  }

  visualization_msgs::msg::Marker makeLineMarker(
    int id, const std::string &ns, const std::vector<double> &xs, const std::vector<double> &ys,
    double width, double r, double g, double b)
  {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id_;
    marker.header.stamp = now();
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = width;
    marker.color.a = 1.0;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.pose.orientation.w = 1.0;

    for (size_t i = 0; i < xs.size(); ++i) {
      geometry_msgs::msg::Point p;
      p.x = xs[i];
      p.y = ys[i];
      p.z = 0.0;
      marker.points.push_back(p);
    }

    return marker;
  }

  std::unique_ptr<ftp::FrenetPlanner<double>> planner_;
  ftp::TrackData track_;
  ftp::Spline2D spline_ref_;
  ftp::Spline2D spline_left_;
  ftp::Spline2D spline_right_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr centerline_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr target_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::TimerBase::SharedPtr centerline_timer_;

  std::string frame_id_;
  size_t lookahead_index_{6};
  int overtake_strategy_{0};
  bool debug_log_{true};
  double last_s_guess_{0.0};

  bool use_test_obstacle_{true};
  bool force_debug_speed_when_stopped_{true};
  double test_obstacle_s_offset_{2.0};
  double test_obstacle_d_offset_{0.0};
  double test_obstacle_radius_{0.20};
  double debug_min_speed_{2.0};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FrenetPlannerNode>());
  rclcpp::shutdown();
  return 0;
}