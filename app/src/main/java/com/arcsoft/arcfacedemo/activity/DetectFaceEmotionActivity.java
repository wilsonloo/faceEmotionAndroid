package com.arcsoft.arcfacedemo.activity;


import android.Manifest;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;

import com.arcsoft.arcfacedemo.R;
import com.arcsoft.arcfacedemo.model.DrawInfo;
import com.arcsoft.arcfacedemo.tflite.Classifier;
import com.arcsoft.arcfacedemo.util.ConfigUtil;
import com.arcsoft.arcfacedemo.util.DrawHelper;
import com.arcsoft.arcfacedemo.util.camera.CameraHelper;
import com.arcsoft.arcfacedemo.util.camera.CameraListener;
import com.arcsoft.arcfacedemo.util.face.RecognizeColor;
import com.arcsoft.arcfacedemo.widget.FaceRectView;
import com.arcsoft.face.AgeInfo;
import com.arcsoft.face.ErrorInfo;
import com.arcsoft.face.Face3DAngle;
import com.arcsoft.face.FaceEngine;
import com.arcsoft.face.FaceInfo;
import com.arcsoft.face.GenderInfo;
import com.arcsoft.face.LivenessInfo;
import com.arcsoft.face.VersionInfo;
import com.arcsoft.face.enums.DetectMode;
import com.arcsoft.face.model.ArcSoftImageInfo;
import com.tencent.bugly.crashreport.CrashReport;

import java.io.ByteArrayOutputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;


public class DetectFaceEmotionActivity extends BaseActivity implements ViewTreeObserver.OnGlobalLayoutListener {
    /**
     * 所需的所有权限信息
     */
    private static final int ACTION_REQUEST_PERMISSIONS = 0x001;
    private static final String[] NEEDED_PERMISSIONS = new String[]{
            Manifest.permission.CAMERA,
            Manifest.permission.READ_PHONE_STATE
    };

    private static final String TAG = "DetectFaceEmoActivity";
    private Integer rgbCameraId = Camera.CameraInfo.CAMERA_FACING_BACK;
    private int processMask = FaceEngine.ASF_AGE | FaceEngine.ASF_FACE3DANGLE | FaceEngine.ASF_GENDER | FaceEngine.ASF_LIVENESS;


    /**
     * 相机预览显示的控件，可为SurfaceView或TextureView
     */
    private View previewView;
    private FaceRectView faceRectView;
    private FaceRectView emotionRectView;

    private Camera.Size previewSize;

    private CameraHelper cameraHelper;
    private DrawHelper drawHelper;

    private FaceEngine faceEngine;
    private int afCode = -1;

    private Classifier mClassifier;
    public Classifier getClassifier(){ return mClassifier;}

    // 私有函数列表 ***********************************************************
    private void initEngine() {
        int combinedMask = FaceEngine.ASF_FACE_DETECT | FaceEngine.ASF_AGE | FaceEngine.ASF_FACE3DANGLE | FaceEngine.ASF_GENDER | FaceEngine.ASF_LIVENESS;
        faceEngine = new FaceEngine();
        afCode = faceEngine.init(this,
                DetectMode.ASF_DETECT_MODE_VIDEO,
                ConfigUtil.getFtOrient(this),
                16,
                20,
                combinedMask);

        VersionInfo versionInfo = new VersionInfo();
        faceEngine.getVersion(versionInfo);
        Log.i(TAG, "initEngine:  init: " + afCode + "  version:" + versionInfo);
        if (afCode != ErrorInfo.MOK) {
            showToast( getString(R.string.init_failed, afCode));
        }
    }

    private void initCamera() {
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);

        CameraListener cameraListener = new CameraListener() {
            @Override
            public void onCameraOpened(Camera camera, int cameraId, int displayOrientation, boolean isMirror) {
                Log.i(TAG, "onCameraOpened: " + cameraId + "  " + displayOrientation + " " + isMirror);
                previewSize = camera.getParameters().getPreviewSize();
                drawHelper = new DrawHelper(previewSize.width,
                        previewSize.height,
                        previewView.getWidth(),
                        previewView.getHeight(),
                        displayOrientation,
                        cameraId,
                        isMirror,
                        false,
                        false);
            }

            @Override
            public void onPreview(byte[] nv21, Camera camera) {
                if (faceRectView != null) {
                    faceRectView.clearFaceInfo();
                }
                List<FaceInfo> faceInfoList = new ArrayList<>();
//                long start = System.currentTimeMillis();
                int code = faceEngine.detectFaces(nv21, previewSize.width, previewSize.height, FaceEngine.CP_PAF_NV21, faceInfoList);
                if (code == ErrorInfo.MOK && faceInfoList.size() > 0) {
                    code = faceEngine.process(nv21, previewSize.width, previewSize.height, FaceEngine.CP_PAF_NV21, faceInfoList, processMask);
                    if (code != ErrorInfo.MOK) {
                        return;
                    }
                } else {
                    return;
                }

                List<AgeInfo> ageInfoList = new ArrayList<>();
                List<GenderInfo> genderInfoList = new ArrayList<>();
                List<Face3DAngle> face3DAngleList = new ArrayList<>();
                List<LivenessInfo> faceLivenessInfoList = new ArrayList<>();
                int ageCode = faceEngine.getAge(ageInfoList);
                int genderCode = faceEngine.getGender(genderInfoList);
                int face3DAngleCode = faceEngine.getFace3DAngle(face3DAngleList);
                int livenessCode = faceEngine.getLiveness(faceLivenessInfoList);

                // 有其中一个的错误码不为ErrorInfo.MOK，return
                if ((ageCode | genderCode | face3DAngleCode | livenessCode) != ErrorInfo.MOK) {
                    return;
                }

                // 摄像机画面对应的位图
                Bitmap previewBitmap = null;
                {
//                   camera.setOneShotPreviewCallback(null);
                    //处理data
                    Camera.Size previewSize = camera.getParameters().getPreviewSize();//获取尺寸,格式转换的时候要用到
                    BitmapFactory.Options newOpts = new BitmapFactory.Options();
                    newOpts.inJustDecodeBounds = true;
                    YuvImage yuvimage = new YuvImage(
                            nv21,
                            ImageFormat.NV21,
                            previewSize.width,
                            previewSize.height,
                            null);
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    yuvimage.compressToJpeg(new Rect(0, 0, previewSize.width, previewSize.height), 100, baos);// 80--JPG图片的质量[0-100],100最高
                    byte[] rawImage = baos.toByteArray();
                    //将rawImage转换成bitmap
                    BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inPreferredConfig = Bitmap.Config.RGB_565;
                    previewBitmap = BitmapFactory.decodeByteArray(rawImage, 0, rawImage.length, options);
                }

                if (faceRectView != null && emotionRectView != null && drawHelper != null) {
                    List<DrawInfo> drawInfoList = new ArrayList<>();
                    for (int i = 0; i < faceInfoList.size(); i++) {
                        // 抓取脸部的位图
                        Rect rawFaceRect = faceInfoList.get(i).getRect();
                        Rect adjustFaceRect = drawHelper.adjustRect(rawFaceRect);
                        int x = Math.max(0, rawFaceRect.left);
                        int y = Math.max(0, rawFaceRect.top);
                        int width = Math.min(previewBitmap.getWidth(), rawFaceRect.right - x);
                        int height = Math.min(previewBitmap.getHeight(), rawFaceRect.bottom - y);
                        Matrix matrix = new Matrix();
                        matrix.setRotate(90);
                        Bitmap faceBitmap = Bitmap.createBitmap(previewBitmap, x, y, width, height, matrix, false);

                        Bundle bundle = new Bundle();
                        // 进行预测和绘制
                        Object[] predictResult = predict(faceBitmap);
                        if(predictResult != null) {
                            String emotionType = (String) predictResult[0];
                            Float confidence = (Float) predictResult[1];

                            bundle.putString("emotionType", emotionType);
                            bundle.putFloat("confidence", confidence);
                            bundle.putInt("emotionResourceId", mClassifier.GetEmotionResourceId(emotionType));
                        }

                        DrawInfo newDrawInfo = new DrawInfo(
                                adjustFaceRect,
                                genderInfoList.get(i).getGender(),
                                ageInfoList.get(i).getAge(),
                                faceLivenessInfoList.get(i).getLiveness(),
                                RecognizeColor.COLOR_UNKNOWN,
                                null,
                                bundle);

                        drawInfoList.add(newDrawInfo);
                    }
                    drawHelper.draw(faceRectView, drawInfoList);
                    drawHelper.draw(emotionRectView, drawInfoList);
                }
            }

            // 进行预测
            private Object[] predict(Bitmap faceBitmap) {
                // 进行分类预测，并产生表情
                Bitmap faceBitmap8888 = faceBitmap.copy(Bitmap.Config.ARGB_8888, true);

                ArrayList<Classifier.Recognition> recognitions = (ArrayList<Classifier.Recognition>) mClassifier.RecognizeImage(faceBitmap8888, 0);
                if (recognitions.size() > 0) {
                    Classifier.Recognition predict = recognitions.get(0);
                    if (predict.getConfidence() > 0.8) {
                        // 获取表镜名称、对应图片资源
                        String emotionType = predict.getTitle();
                        Float confidence = predict.getConfidence();

                        return new Object[]{emotionType, confidence};
                    }
                }

                return null;
            }

            @Override
            public void onCameraClosed() {
                Log.i(TAG, "onCameraClosed: ");
            }

            @Override
            public void onCameraError(Exception e) {
                Log.i(TAG, "onCameraError: " + e.getMessage());
            }

            @Override
            public void onCameraConfigurationChanged(int cameraID, int displayOrientation) {
                if (drawHelper != null) {
                    drawHelper.setCameraDisplayOrientation(displayOrientation);
                }
                Log.i(TAG, "onCameraConfigurationChanged: " + cameraID + "  " + displayOrientation);
            }
        };

        cameraHelper = new CameraHelper.Builder()
                .previewViewSize(new Point(previewView.getMeasuredWidth(), previewView.getMeasuredHeight()))
                .rotation(getWindowManager().getDefaultDisplay().getRotation())
                .specificCameraId(rgbCameraId != null ? rgbCameraId : Camera.CameraInfo.CAMERA_FACING_FRONT)
                .isMirror(false)
                .previewOn(previewView)
                .cameraListener(cameraListener)
                .build();
        cameraHelper.init();
        cameraHelper.start();
    }

    private void initClassifier(int numThreads){
        try {
            mClassifier = Classifier.Create(this,
                    Classifier.Model.FLOAT_MOBILENET,
                    Classifier.Device.CPU,
                    numThreads);

        } catch (Exception e) {
            e.printStackTrace();
            CrashReport.postCatchedException(e);
        }
    }

    // 函数列表 ***********************************************************
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 摄像头
        Intent intent = getIntent();
        rgbCameraId = intent.getIntExtra("whichCamera", Camera.CameraInfo.CAMERA_FACING_BACK);

        setContentView(R.layout.activity_detect_face_emotion);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
            WindowManager.LayoutParams attributes = getWindow().getAttributes();
            attributes.systemUiVisibility = View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION;
            getWindow().setAttributes(attributes);
        }

        // Activity启动后就锁定为启动时的方向
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LOCKED);

        previewView = findViewById(R.id.fe_texture_preview);
        faceRectView = findViewById(R.id.fe_face_rect_view);
        emotionRectView = findViewById(R.id.fe_emotion_rect_view);

        // 分类器
        initClassifier(4);

        //在布局结束后才做初始化操作
        previewView.getViewTreeObserver().addOnGlobalLayoutListener(this);
    }

    @Override
    void afterRequestPermission(int requestCode, boolean isAllGranted) {

    }

    /**
     * 在{@link #previewView}第一次布局完成后，去除该监听，并且进行引擎和相机的初始化
     */
    @Override
    public void onGlobalLayout() {
        previewView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
        if (!checkPermissions(NEEDED_PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, NEEDED_PERMISSIONS, ACTION_REQUEST_PERMISSIONS);
        } else {
            initEngine();
            initCamera();
        }
    }
}
