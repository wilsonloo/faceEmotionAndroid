package com.arcsoft.arcfacedemo.tfpb;

import android.graphics.Bitmap;
import android.os.AsyncTask;

import com.arcsoft.arcfacedemo.activity.BaseActivity;
import com.tencent.bugly.crashreport.CrashReport;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class ClassifierPb {
    private static final String MODEL_PATH = "file:///android_asset/saved_model.pb";
    private static final String INPUT_NAME = "input_1";
    private static final String OUTPUT_NAME = "output_1";

    private BaseActivity activity;
    private TensorFlowInferenceInterface tf;

    // 预处理例如图像的像素输入数据要进行归一化成[0.0, 1.0]
    private float[] floatValues;
    private float[] PREDICTIONS = new float[1000];

    //FUNCTION TO COMPUTE THE MAXIMUM PREDICTION AND ITS CONFIDENCE
    public static Object[] argmax(float[] array) {
        int best = -1;
        float best_confidence = 0.0f;

        for (int i = 0; i < array.length; i++) {
            float value = array[i];
            if (value > best_confidence) {
                best_confidence = value;
                best = i;
            }
        }

        return new Object[]{best, best_confidence};
    }

    public ClassifierPb(BaseActivity activity) {
        this.activity = activity;

        // 加载模型
        this.tf = new TensorFlowInferenceInterface(activity.getAssets(), MODEL_PATH);
    }

    public void Predict(Bitmap bitmap) {
        System.out.println(5555);
        // 使用后台县城进行预测
        new AsyncTask<Integer, Integer, Integer>() {
            @Override
            protected Integer doInBackground(Integer... integers) {
                // 调整图片尺寸为 192*192
                Bitmap resized_image = ImageUtils.processBitmap(bitmap, 192);

                // 对像素进行标准化
                floatValues = ImageUtils.normalizeBitmap(resized_image, 224, 127.5f, 1.0f);

                // 填充输入数据到模型
                tf.feed(INPUT_NAME, floatValues, 192, 192, 3);

                // 计算预测值
                tf.run(new String[]{OUTPUT_NAME});

                // 转存置信度的概率分布图
                tf.fetch(OUTPUT_NAME, PREDICTIONS);

                // 获取置信度最高的分类
                Object[] topResult = argmax(PREDICTIONS);
                int class_index = (Integer) topResult[0];
                float confidence = (Float) topResult[1];

                try {
                    // 置信度文本
                    final String conf = String.valueOf(confidence * 100).substring(0, 5);

                    // 获取类别的标签名
                    final String label = ImageUtils.getLabel(activity.getAssets().open("labels.txt"), class_index);

                    System.out.printf(">>>>>>>>> %s : %s\n", label, conf);

                } catch (Exception e) {
                    CrashReport.postCatchedException(e);
                }

                return 0;
            }
        }.execute(0);
    }
}
