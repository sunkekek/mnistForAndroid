package com.example.pytorchforandroid;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.Manifest;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;


public class MainActivity extends AppCompatActivity implements View.OnClickListener{
    public static final int PERMISSION_REQUEST_CODE = 0;
    public static final int TAKE_PHOTO_REQUEST_CODE = 0;
    private final String TAG = "MainActivity_sunkeke";

    private Uri imageUri;
    private Bitmap bitmap;
    private Button openCameraButton;
    private ImageView resultImage;
    private TextView resultText;
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.main_activity);

        openCameraButton = findViewById(R.id.open_camera);
        resultImage = findViewById(R.id.get_image);
        resultText = findViewById(R.id.get_text);

        findViewById(R.id.open_camera).setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {
        if (view.getId() == R.id.open_camera) {
            requestPermission();
        }
    }

    /**
     * 申请动态权限
     */
    private void requestPermission() {
        if (ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CODE);
        }else {
            takePhoto();
        }
    }

    /**
     * 用户选择是否开启权限操作后的回调；TODO 同意/拒绝
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // TODO 用户同意开启权限，打开相机
                takePhoto();
            }else{
                Log.d("skk", "权限申请拒绝!");
            }
        }
    }

    /**
     * 打开相机，选择图片
     */
    private void takePhoto() {
        Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        // 确保有一个活动来处理意图
        if (takePhotoIntent.resolveActivity(getPackageManager()) != null) {
            // 创建保存图片的文件夹
            File imageFile = null;
            try {
                imageFile = createImageFile();
            }catch (Exception e){
                e.printStackTrace();
            }
            if (imageFile != null) {
                //TODO imageUri 用来接收拍摄的这张照片的真实路径
                imageUri = FileProvider.getUriForFile(this, "com.example.pytorchforandroid.fileprovider", imageFile);
            }

            takePhotoIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
            startActivityForResult(takePhotoIntent, TAKE_PHOTO_REQUEST_CODE);
        }
    }

    /**
     * 创建一个存放拍的照片的文件
     */
    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
                .format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        Log.d("skk", imageFileName);
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        return File.createTempFile(
                imageFileName,  /* prefix */
                ".bmp",         /* suffix */
                storageDir      /* directory */
        );
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == TAKE_PHOTO_REQUEST_CODE) {
            if (resultCode == Activity.RESULT_OK) {
                try {
                    InputStream inputStream = getContentResolver().openInputStream(imageUri);
                    Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                    long beginTime = System.currentTimeMillis();
                    String identifyResult = getPytorchResult(bitmap);
                    long endTime = System.currentTimeMillis();
                    Log.i("sunkeke_time", "pytorch for android spend time: " + (endTime - beginTime) + "ms");
                    Log.i(TAG + "sunkeke", "identifyResult: " + identifyResult);

                    resultText.setText(identifyResult +"！识别用时：" +String.valueOf(endTime - beginTime) + "ms");
                    resultText.setGravity(Gravity.TOP);
                    resultImage.setImageBitmap(bitmap);
                    openCameraButton.setGravity(Gravity.BOTTOM);


                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    Module module_ori = null;

    private String getPytorchResult(Bitmap bitmap) {
        Log.i(TAG+"sunkeke", "getPytorchResult begin! ");
        try {
            String modelPath = assetFilePath(this, "lenet.ptl");
            Log.i(TAG, "modelPath: " + modelPath);
            module_ori = Module.load(modelPath);
            Log.i(TAG, "load .ptl model success! ");
        } catch (Exception e) {
            Log.e(TAG, "Error reading assets", e);
            e.printStackTrace();
            finish();
        }

        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        Tensor outputTensor = module_ori.forward(IValue.from(inputTensor)).toTensor();
        final float[] scores = outputTensor.getDataAsFloatArray();
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIndex = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIndex = i;
            }
        }
        String className = MnistClass.MNIST_CLASSED[maxScoreIndex];
        Log.i(TAG, "className: " + className);
        return className;
    }

    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return  file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}
