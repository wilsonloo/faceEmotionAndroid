package com.arcsoft.arcfacedemo.activity;

import android.os.Bundle;
import android.view.WindowManager;

import com.arcsoft.arcfacedemo.R;
import com.arcsoft.arcfacedemo.faceserver.FaceServer;
import com.arcsoft.arcfacedemo.widget.ProgressDialog;

import java.util.concurrent.Executors;

public class DetectFaceEmotion extends BaseActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detect_face_emotion);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
//        executorService = Executors.newSingleThreadExecutor();
//        tvNotificationRegisterResult = findViewById(R.id.notification_register_result);
//        progressDialog = new ProgressDialog(this);
        FaceServer.getInstance().init(this);
    }

    @Override
    void afterRequestPermission(int requestCode, boolean isAllGranted) {

    }
}
