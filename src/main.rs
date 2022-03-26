// OpenCV____Face and Eye Detection

use opencv::{core, highgui, imgproc, objdetect, prelude::*, types, videoio, Result};

fn main() -> Result<()> {
    let mut camera = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let xml = "haarcascade_eye.xml";
    let mut face_detector = objdetect::CascadeClassifier::new(xml)?;
    let mut img = Mat::default();

    loop {
        camera.read(&mut img)?;
        let mut gray = Mat::default();
        imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        let mut faces = types::VectorOfRect::new();
        face_detector.detect_multi_scale(
            &gray,
            &mut faces,
            1.1,
            10,
            objdetect::CASCADE_SCALE_IMAGE,
            core::Size::new(30, 30),
            core::Size::new(0, 0),
        )?;
        println!("{faces:#?}");
        if faces.len() > 0 {
            for face in faces.iter() {
                imgproc::rectangle(
                    &mut img,
                    face,
                    core::Scalar::new(0f64, 255f64, 0f64, 0f64),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;
            }
        }
        highgui::imshow("grey", &img)?;
        highgui::wait_key(1);
    }

    Ok(())
}
