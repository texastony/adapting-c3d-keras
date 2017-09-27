//! [includes]
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>
//! [includes]

//! [namespace]
using namespace cv;
//! [namespace]

using namespace std;

int main( int argc, char** argv)
{
  //! [load]
  String imageName( "../../../../../Desktop/Calton-Hill-16-9.jpg" ); // by default
  if (argc > 1)
  {
    imageName = argv[1];
  }
  //! [load]

  //! [mat]
  Mat image;
  //! [mat]

  //! [imread]
  image = imread( imageName, IMREAD_COLOR);
  //! [imread]

  if (image.empty())
  {
    cout << "Could not open or find image" << std::endl;
    return -1;
  }

  //! [window]
  namedWindow( "Display window", WINDOW_AUTOSIZE);
  //! [window]
  
  //! [imshow]
  imshow( "Display window", image);
  //! [imshow]


  //! [wait]
  waitKey(0);
  //! [wait]
  return 0;
}
