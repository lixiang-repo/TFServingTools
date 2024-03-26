import com.google.protobuf.ByteString;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.parallelStream.Collectors;
import java.util.stream.Collectors;

public class Application {

    static List<String> schema;

    static PredictionServiceGrpc.PredictionServiceBlockingStub stub;

    public static void main(String[] args) throws IOException {
        initSchema("schema.conf");
        stub = getPredictionServiceBlockingStub("60.205.130.26", 8500);

        //PredictRequest
        Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
        modelSpecBuilder.setName("model");
        modelSpecBuilder.setSignatureName("pred");
        predictRequestBuilder.setModelSpec(modelSpecBuilder);

        List<String> input = Arrays.asList("0", "1");
        for (String name : schema) {
            predictRequestBuilder.putInputs(name, buildStringTensorProto(input));
        }
        List<Float> labels = Arrays.asList((float) 0, (float) 0);
        predictRequestBuilder.putInputs("label", buildFloatTensorProto(labels));

        for (int i = 0;i < 1e5;i++) {
            long time1 = System.currentTimeMillis();
            Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
            Map<String, TensorProto> result = predictResponse.getOutputsMap();
            List<Float> scores1 = result.get("output1").getFloatValList();
            long time2 = System.currentTimeMillis();

//            System.out.println(result);
            System.out.println("scores1: " + scores1 + " time: " + (time2 - time1));
//            break;
        }

    }
    public static void initSchema(String schemaConf) throws IOException {
        schema = new ArrayList<>();

        InputStream fileInputStream = Application.class.getClassLoader().getResourceAsStream(schemaConf);
        BufferedReader br = new BufferedReader(new InputStreamReader(fileInputStream));
        String line;
        while ((line = br.readLine()) != null) {
            if (line.startsWith("#") || line.startsWith("label")) {
                continue;
            }
            String name = line.split(" +")[0];
            schema.add(name);
        }
        System.out.println("schema: " + schema);
    }

    public static TensorProto buildIn64TensorProto(List<Long> input) {
        TensorProto.Builder inputTensorProto = TensorProto.newBuilder();
        inputTensorProto.setDtype(DataType.DT_INT64);
        inputTensorProto.addAllInt64Val(input);
        TensorShapeProto.Builder inputShapeBuilder = TensorShapeProto.newBuilder();
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(input.size()));
        inputTensorProto.setTensorShape(inputShapeBuilder.build());

        return inputTensorProto.build();
    }

    public static TensorProto buildIn64ListTensorProto(List<List<Long>> inputs) {
        TensorProto.Builder inputTensorProto = TensorProto.newBuilder();
        inputTensorProto.setDtype(DataType.DT_INT64);
        for (List<Long> input : inputs) {
            inputTensorProto.addAllInt64Val(input);
        }
        TensorShapeProto.Builder inputShapeBuilder = TensorShapeProto.newBuilder();
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(inputs.size()));
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(inputs.get(0).size()));
        inputTensorProto.setTensorShape(inputShapeBuilder.build());

        return inputTensorProto.build();
    }

    public static TensorProto buildStringTensorProto(List<String> input) {
        TensorProto.Builder inputTensorProto = TensorProto.newBuilder();
        inputTensorProto.setDtype(DataType.DT_STRING);
        List<ByteString> list = input.parallelStream().map(x -> ByteString.copyFromUtf8(x)).collect(Collectors.toList());
        inputTensorProto.addAllStringVal(list);
        TensorShapeProto.Builder inputShapeBuilder = TensorShapeProto.newBuilder();
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(input.size()));
        inputTensorProto.setTensorShape(inputShapeBuilder.build());

        return inputTensorProto.build();
    }

    public static TensorProto buildStringListTensorProto(List<List<String>> inputs) {
        TensorProto.Builder inputTensorProto = TensorProto.newBuilder();
        inputTensorProto.setDtype(DataType.DT_STRING);
        for (List<String> input : inputs) {
            List<ByteString> list = input.parallelStream().map(x -> ByteString.copyFromUtf8(x)).collect(Collectors.toList());
            inputTensorProto.addAllStringVal(list);
        }
        TensorShapeProto.Builder inputShapeBuilder = TensorShapeProto.newBuilder();
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(inputs.size()));
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(inputs.get(0).size()));
        inputTensorProto.setTensorShape(inputShapeBuilder.build());

        return inputTensorProto.build();
    }

    public static TensorProto buildFloatTensorProto(List<Float> input) {
        TensorProto.Builder inputTensorProto = TensorProto.newBuilder();
        inputTensorProto.setDtype(DataType.DT_FLOAT);
        inputTensorProto.addAllFloatVal(input);
        TensorShapeProto.Builder inputShapeBuilder = TensorShapeProto.newBuilder();
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(input.size()));
        inputTensorProto.setTensorShape(inputShapeBuilder.build());

        return inputTensorProto.build();
    }

    public static TensorProto buildFloatListTensorProto(List<List<Float>> inputs) {
        TensorProto.Builder inputTensorProto = TensorProto.newBuilder();
        inputTensorProto.setDtype(DataType.DT_FLOAT);
        for (List<Float> list : inputs) {
            inputTensorProto.addAllFloatVal(list);
        }
        TensorShapeProto.Builder inputShapeBuilder = TensorShapeProto.newBuilder();
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(inputs.size()));
        inputShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(inputs.get(0).size()));
        inputTensorProto.setTensorShape(inputShapeBuilder.build());

        return inputTensorProto.build();
    }

    public static PredictionServiceGrpc.PredictionServiceBlockingStub getPredictionServiceBlockingStub(String ip, int port) {
        ManagedChannel channel = ManagedChannelBuilder.forAddress(ip, port).usePlaintext(true).build();
        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
        return stub;
    }



}
