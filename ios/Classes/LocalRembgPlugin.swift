import Flutter
import UIKit
import Vision

@available(iOS 15.0, *)
public class LocalRembgPlugin: NSObject, FlutterPlugin {

    private var segmentationRequest: VNGeneratePersonSegmentationRequest?

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "methodChannel.localRembg", binaryMessenger: registrar.messenger())
        let instance = LocalRembgPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        segmentationRequest = VNGeneratePersonSegmentationRequest()
        segmentationRequest?.qualityLevel = .accurate
        segmentationRequest?.outputPixelFormat = kCVPixelFormatType_OneComponent8

        switch call.method {
        case "removeBackground":
            if isRunningOnSimulator() {
                result(["status": 0, "message": "Please use a real device"])
                return
            }

            guard let arguments = call.arguments as? [String: Any],
                  let shouldCropImage = arguments["cropImage"] as? Bool else {
                result(["status": 0, "message": "Invalid arguments"])
                return
            }

            var image: UIImage?
            if let imagePath = arguments["imagePath"] as? String {
                image = UIImage(contentsOfFile: imagePath)
            } else if let defaultImageUint8List = arguments["imageUint8List"] as? FlutterStandardTypedData {
                image = UIImage(data: defaultImageUint8List.data)
            }

            guard let loadedImage = image else {
                result(["status": 0, "message": "Unable to load image"])
                return
            }

            applyFilter(image: loadedImage, shouldCropImage: shouldCropImage) { [self] resultImage, numFaces in
                guard let resultImage = resultImage else {
                    result(["status": 0, "message": "Unable to process image"])
                    return
                }

                if let imageData = resultImage.pngData() {
                    result(["status": 1, "message": "Success", "imageBytes": FlutterStandardTypedData(bytes: imageData)])
                } else {
                    result(["status": 0, "message": "Unable to convert image to bytes"])
                }
            }

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // Helper function to check if running on the simulator
    private func isRunningOnSimulator() -> Bool {
        return TARGET_OS_SIMULATOR != 0
    }

    // Apply the segmentation request to filter out the background
    private func applyFilter(image: UIImage, shouldCropImage: Bool, completion: @escaping (UIImage?, Int) -> Void) {
        guard let cgImage = image.cgImage else {
            completion(nil, 0)
            return
        }

        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        DispatchQueue.global().async {
            do {
                try requestHandler.perform([self.segmentationRequest!])
                if let mask = self.segmentationRequest?.results?.first?.pixelBuffer {
                    let maskImage = self.createMaskImage(from: mask)

                    // Check if foreground is present, if not return original image
                    let isForegroundPresent = self.detectForeground(in: mask)
                    if !isForegroundPresent {
                        completion(image, 0)
                        return
                    }

                    self.applyBackgroundMask(maskImage, image: image, shouldCropImage: shouldCropImage, completion: completion)
                } else {
                    completion(nil, 0)
                }
            } catch {
                completion(nil, 0)
            }
        }
    }

    // Create a mask image from the pixel buffer returned by the segmentation request
    private func createMaskImage(from pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        return context.createCGImage(ciImage, from: ciImage.extent)
    }

    // Apply the background mask to the original image to remove the background
    private func applyBackgroundMask(_ maskImage: CGImage?, image: UIImage, shouldCropImage: Bool, completion: @escaping (UIImage?, Int) -> Void) {
        guard let maskImage = maskImage, let cgImage = image.cgImage else {
            completion(nil, 0)
            return
        }

        let mainImage = CIImage(cgImage: cgImage)
        let maskCI = CIImage(cgImage: maskImage)

        let filter = CIFilter(name: "CIBlendWithMask")
        filter?.setValue(mainImage, forKey: kCIInputImageKey)
        filter?.setValue(maskCI, forKey: kCIInputMaskImageKey)

        if let outputImage = filter?.outputImage {
            let context = CIContext()
            if let cgOutputImage = context.createCGImage(outputImage, from: outputImage.extent) {
                let finalImage = UIImage(cgImage: cgOutputImage)
                completion(finalImage, 1)
            } else {
                completion(nil, 0)
            }
        } else {
            completion(nil, 0)
        }
    }

    // Detect if there's a significant foreground in the mask
    private func detectForeground(in mask: CVPixelBuffer) -> Bool {
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }

        let width = CVPixelBufferGetWidth(mask)
        let height = CVPixelBufferGetHeight(mask)

        guard let baseAddress = CVPixelBufferGetBaseAddress(mask) else { return false }

        let buffer = UnsafeMutablePointer<UInt8>(baseAddress.assumingMemoryBound(to: UInt8.self))

        // Simple logic: if enough non-background pixels are detected, return true
        let threshold = 0.1 * Double(width * height)
        var nonBackgroundPixelCount = 0

        for y in 0..<height {
            for x in 0..<width {
                let pixelValue = buffer[y * width + x]
                if pixelValue > 50 { // Threshold to determine foreground
                    nonBackgroundPixelCount += 1
                }
            }
        }

        return Double(nonBackgroundPixelCount) > threshold
    }
}
