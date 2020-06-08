// Generated by gencpp from file apriltags2_ros/AnalyzeSingleImage.msg
// DO NOT EDIT!


#ifndef APRILTAGS2_ROS_MESSAGE_ANALYZESINGLEIMAGE_H
#define APRILTAGS2_ROS_MESSAGE_ANALYZESINGLEIMAGE_H

#include <ros/service_traits.h>


#include <apriltags2_ros/AnalyzeSingleImageRequest.h>
#include <apriltags2_ros/AnalyzeSingleImageResponse.h>


namespace apriltags2_ros
{

struct AnalyzeSingleImage
{

typedef AnalyzeSingleImageRequest Request;
typedef AnalyzeSingleImageResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct AnalyzeSingleImage
} // namespace apriltags2_ros


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::apriltags2_ros::AnalyzeSingleImage > {
  static const char* value()
  {
    return "d60d994450f73cbdba772751d78c9952";
  }

  static const char* value(const ::apriltags2_ros::AnalyzeSingleImage&) { return value(); }
};

template<>
struct DataType< ::apriltags2_ros::AnalyzeSingleImage > {
  static const char* value()
  {
    return "apriltags2_ros/AnalyzeSingleImage";
  }

  static const char* value(const ::apriltags2_ros::AnalyzeSingleImage&) { return value(); }
};


// service_traits::MD5Sum< ::apriltags2_ros::AnalyzeSingleImageRequest> should match
// service_traits::MD5Sum< ::apriltags2_ros::AnalyzeSingleImage >
template<>
struct MD5Sum< ::apriltags2_ros::AnalyzeSingleImageRequest>
{
  static const char* value()
  {
    return MD5Sum< ::apriltags2_ros::AnalyzeSingleImage >::value();
  }
  static const char* value(const ::apriltags2_ros::AnalyzeSingleImageRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::apriltags2_ros::AnalyzeSingleImageRequest> should match
// service_traits::DataType< ::apriltags2_ros::AnalyzeSingleImage >
template<>
struct DataType< ::apriltags2_ros::AnalyzeSingleImageRequest>
{
  static const char* value()
  {
    return DataType< ::apriltags2_ros::AnalyzeSingleImage >::value();
  }
  static const char* value(const ::apriltags2_ros::AnalyzeSingleImageRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::apriltags2_ros::AnalyzeSingleImageResponse> should match
// service_traits::MD5Sum< ::apriltags2_ros::AnalyzeSingleImage >
template<>
struct MD5Sum< ::apriltags2_ros::AnalyzeSingleImageResponse>
{
  static const char* value()
  {
    return MD5Sum< ::apriltags2_ros::AnalyzeSingleImage >::value();
  }
  static const char* value(const ::apriltags2_ros::AnalyzeSingleImageResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::apriltags2_ros::AnalyzeSingleImageResponse> should match
// service_traits::DataType< ::apriltags2_ros::AnalyzeSingleImage >
template<>
struct DataType< ::apriltags2_ros::AnalyzeSingleImageResponse>
{
  static const char* value()
  {
    return DataType< ::apriltags2_ros::AnalyzeSingleImage >::value();
  }
  static const char* value(const ::apriltags2_ros::AnalyzeSingleImageResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // APRILTAGS2_ROS_MESSAGE_ANALYZESINGLEIMAGE_H