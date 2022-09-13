package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/3d0c/gmf"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
	"golang.org/x/image/draw"
)

var (
	flagInput         = flag.String("i", "", "the input path")
	flagStreamN       = flag.Int("s", -1, "stream number")
	flagVerbose       = flag.Bool("v", false, "verbose output")
	flagDebug         = flag.Bool("d", false, "ffmpeg debug loglevel")
	flagTimer         = flag.Bool("t", false, "print out elapsed time at the end")
	flagSaveImage     = flag.Bool("h", false, "save histogram image")
	flagMagnitudeMult = flag.Float64("m", 0.07, "max magnitude multipler used for max hz value calculation")
	flagFFTWinSize    = flag.Int("f", 2048, "FFT windows size, must be a power of two")
	sampleFmt         = gmf.AV_SAMPLE_FMT_S32
	sampleSize        = 4
	FFTHalfWinSize    = *flagFFTWinSize / 2
)

const ()

type stream struct {
	decCodec        *gmf.Codec
	decCodecContext *gmf.CodecCtx
	inputStream     *gmf.Stream
}

func main() {
	start := time.Now()

	// Parse flags
	flag.Parse()

	// Adjust flags after parsing
	FFTHalfWinSize = *flagFFTWinSize / 2
	if *flagDebug {
		*flagVerbose = true
	}

	// Usage
	if *flagInput == "" {
		fmt.Println("Usage: getfreqy -i <input path> [options]")
		flag.PrintDefaults()
		return
	}

	// Set ffmpeg log level
	gmf.LogSetLevel(gmf.AV_LOG_FATAL)
	if *flagDebug {
		gmf.LogSetLevel(gmf.AV_LOG_DEBUG)
	}

	// Get audio samples as floats
	floatSamples, err := resampleInputToFloatSamples(*flagInput)
	if err != nil {
		panic(err)
	}

	// Get spectrogram values from FFT of audio samples
	spectrogram, err := buildSpectrogram(floatSamples)
	if err != nil {
		panic(err)
	}

	hzMax, hzMaxRow := getSuitableHzValueAndRow(spectrogram)

	fmt.Printf("%v\n", hzMax)

	if *flagSaveImage {
		basename := filepath.Base(*flagInput)
		outputPath := strings.TrimSuffix(basename, filepath.Ext(basename)) + ".png"

		fmt.Printf("saving %q...\n", outputPath)
		err = saveToPng(visualizeSpectre(spectrogram, hzMaxRow), outputPath)
		if err != nil {
			panic(err)
		}
	}

	if *flagTimer {
		fmt.Printf("%v\n", time.Since(start))
	}
}

func getSuitableHzValueAndRow(spectrogram [][]float64) (hzMax, hzMaxRow int) {
	// Get number of rows and columns in spectrogram
	numRows := len(spectrogram)
	var numCols int
	if numRows > 0 {
		numCols = len(spectrogram[0])
	}

	// Get min and max magnitude values in spectrogram
	minInSpectre, maxInSpectre := math.MaxFloat64, float64(0)
	for y := 0; y < numRows; y++ {
		for x := 0; x < numCols; x++ {
			minInSpectre = math.Min(minInSpectre, float64(spectrogram[y][x]))
			maxInSpectre = math.Max(maxInSpectre, float64(spectrogram[y][x]))
		}
	}

	// Highest sum of magnituteds in a hz window row
	magSumMax := math.SmallestNonzeroFloat64

	if len(spectrogram) > 0 {
		// Find highest sum of magnituteds in a hz window row
		for row := numRows - 1; row >= 0; row-- {
			magSum := 0.0
			for col := 0; col < numCols; col++ {
				mag := spectrogram[row][col]
				mag -= float64(minInSpectre)
				magSum += mag
			}

			if magSum > magSumMax {
				magSumMax = magSum
			}
		}

		//
		for row := numRows - 1; row >= 0; row-- {
			magSum := 0.0
			for col := 0; col < numCols; col++ {
				mag := spectrogram[row][col]
				mag -= float64(minInSpectre)
				magSum += mag
			}

			if magSum/float64(numCols) > magSumMax/float64(numCols)**flagMagnitudeMult {
				hzMax = int(math.Round(float64(48000/2) / float64(numRows) * float64(row)))
				hzMaxRow = row
				break
			}
		}
	}

	return
}

func resampleInputToFloatSamples(input string) ([]float64, error) {
	// Alloc input format context
	inputFormatContext, err := gmf.NewInputCtx(input)
	if err != nil {
		return nil, err
	}
	defer inputFormatContext.Free()

	srcStreams := map[int]*gmf.Stream{}
	firstAudioStreamN := -1

	// Loop through streams
	for idx := 0; idx < inputFormatContext.StreamsCnt(); idx++ {
		stream, err := inputFormatContext.GetStream(idx)
		if err != nil {
			return nil, fmt.Errorf("error getting stream - %s", err)
		}

		// Only process audio
		if !stream.IsAudio() {
			continue
		}

		if firstAudioStreamN < 0 {
			firstAudioStreamN = idx
		}

		srcStreams[idx] = stream
	}

	// Sort stream map
	keys := make([]int, 0, len(srcStreams))
	for k := range srcStreams {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	// Print out streams info
	if *flagVerbose {
		for _, i := range keys {
			stream := srcStreams[i]
			fmt.Printf("0:%d Audio: %s, %v Hz, %v, %v, %v kb/s\n", i, stream.CodecCtx().Codec().Name(), stream.CodecCtx().SampleRate(), stream.CodecCtx().GetChannelLayoutName(), stream.CodecCtx().GetSampleFmtName(), stream.CodecCtx().BitRate()/1000)
		}
	}

	// Select the first audio stream if stream number is less then zero
	if *flagStreamN < 0 {
		*flagStreamN = firstAudioStreamN
	}
	if *flagVerbose {
		fmt.Printf("stream %v selected\n", *flagStreamN)
	}

	// Select stream
	inStream, ok := srcStreams[*flagStreamN]
	if !ok {
		return nil, fmt.Errorf("%v is not an audio stream", *flagStreamN)
	}
	inCodecCtx := inStream.CodecCtx()

	// Resample options
	options := []*gmf.Option{
		{Key: "in_channel_count", Val: inCodecCtx.Channels()},
		{Key: "out_channel_count", Val: 1},
		{Key: "in_sample_rate", Val: inCodecCtx.SampleRate()},
		{Key: "out_sample_rate", Val: 48000},
		{Key: "in_sample_fmt", Val: inCodecCtx.SampleFmt()},
		{Key: "out_sample_fmt", Val: sampleFmt},
	}

	// Create swresample context
	swrCtx, err := gmf.NewSwrCtx(options, inCodecCtx.Channels(), inCodecCtx.SampleFmt())
	if err != nil {
		return nil, fmt.Errorf("new swr context error: %v", err)
	}
	if swrCtx == nil {
		return nil, fmt.Errorf("unable to create Swr Context")
	}

	// Loop through packets
	var floatSamples []float64
	for packet := range inputFormatContext.GetNewPackets() {
		// Skip packets from other streams
		if packet.StreamIndex() != *flagStreamN {
			continue
		}

		// Decode packet
		inFrame, ret := inStream.CodecCtx().Decode2(packet)
		packet.Free()
		if ret < 0 {
			return nil, fmt.Errorf("%v", gmf.AvError(ret))
		}

		// Resample packet with swresample
		outFrame, err := swrCtx.Convert(inFrame)
		if err != nil {
			fmt.Printf("convert audio error: %v", err)
			break
		}
		if outFrame == nil {
			continue
		}
		inFrame.Free()

		// Set output channel number (not necessary)
		outFrame.SetChannels(1)

		// Get raw audio samples as floats
		floatSamples = append(floatSamples, rawAudioToFloatSamples(outFrame.GetRawAudioData(0), sampleSize)...)
	}

	return floatSamples, nil
}

func rawAudioToFloatSamples(rawAudio []byte, sampleSize int) (output []float64) {
	uints32 := make([]uint32, len(rawAudio)/sampleSize)
	for i := 0; i+sampleSize <= len(rawAudio); i += sampleSize {
		uints32[i/sampleSize] = binary.LittleEndian.Uint32(rawAudio[i : i+sampleSize])
	}

	return normalize(uints32)
}

func normalize(input []uint32) []float64 {
	normalized := make([]float64, len(input))

	for i, val := range input {
		x := float64(int32(val))
		normalized[i] = x / math.MaxUint32
	}

	return normalized
}

func buildSpectrogram(wave []float64) (spectrogram [][]float64, err error) {
	waveLen := len(wave)
	if waveLen == 0 {
		return
	}

	winFunc := window.Hann(*flagFFTWinSize + 2)[1 : *flagFFTWinSize+1]

	spectrogram = make([][]float64, FFTHalfWinSize+1) // rows x cols

	win := make([]float64, *flagFFTWinSize)
	winZeroes := make([]float64, *flagFFTWinSize)

	stride := *flagFFTWinSize - FFTHalfWinSize
	winCnt := (waveLen + stride - 1) / stride
	for winIdx, offs := 0, 0; winIdx < winCnt; winIdx, offs = winIdx+1, offs+stride {
		idx := minInt(waveLen, offs+*flagFFTWinSize)
		if idx < (offs + *flagFFTWinSize) {
			copy(win, winZeroes)
		}

		for i := offs; i < idx; i++ {
			win[i-offs] = float64(wave[i])
		}

		for i, w := range winFunc {
			win[i] *= w
		}

		line := fft.FFTReal(win)
		for i := 0; i < FFTHalfWinSize+1; i++ {
			win[i] = cmplx.Abs(line[i])
		}

		for i, mag := range win[0 : FFTHalfWinSize+1] {
			spectrogram[i] = append(spectrogram[i], float64(mag))
		}
	}

	spectreMin, spectreMax := float64(math.MaxFloat32), float64(0.0)
	for _, line := range spectrogram {
		for _, mag := range line {
			spectreMax = maxFloat(spectreMax, mag)
			spectreMin = minFloat(spectreMin, mag)
		}
	}

	if spectreMax < 1e-6 {
		return nil, fmt.Errorf("zero signal")
	}

	minMag := spectreMax / 1e6
	spectreMean, cnt := float64(0), 0
	for y, line := range spectrogram {
		for x, magRaw := range line {
			if magRaw < minMag {
				magRaw = minMag
			}
			mag := math.Log(float64(magRaw))
			spectrogram[y][x] = mag
			spectreMean += mag
			cnt++
		}
	}

	spectreMean /= float64(cnt)

	for y, line := range spectrogram {
		for x := range line {
			spectrogram[y][x] -= float64(spectreMean)
		}
	}

	spectrogram = spectrogram[:len(spectrogram)-1]

	return
}

func minInt(a, b int) int {
	if a < b {
		return a
	} else {
		return b
	}
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	} else {
		return b
	}
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	} else {
		return b
	}
}

func visualizeSpectre(spectre [][]float64, redLineRow int) image.Image {
	numRows := len(spectre)
	var numCols int
	if numRows > 0 {
		numCols = len(spectre[0])
	}

	minInSpectre, maxInSpectre := math.MaxFloat64, float64(0)
	for y := 0; y < numRows; y++ {
		for x := 0; x < numCols; x++ {
			minInSpectre = math.Min(minInSpectre, float64(spectre[y][x]))
			maxInSpectre = math.Max(maxInSpectre, float64(spectre[y][x]))
		}
	}
	diffM := maxInSpectre - minInSpectre

	img := image.NewRGBA(image.Rect(0, 0, numCols, numRows))

	if len(spectre) > 0 {
		for row := 0; row < numRows; row++ {
			for col := 0; col < numCols; col++ {
				vol := spectre[row][col]
				vol -= float64(minInSpectre)

				clr := uint8((255 * vol / float64(diffM)) / 2)
				pix := color.RGBA{clr, clr, clr, 255}

				if row == redLineRow {
					pix = color.RGBA{255, 0, 0, 255}
				}

				x := col
				y := numRows - row - 1

				img.Set(x, y, pix)
			}
		}
	}

	// Downscale image if it's too big
	if img.Bounds().Max.X > 3840 {
		dst := image.NewRGBA(image.Rect(0, 0, img.Bounds().Max.X/2, img.Bounds().Max.Y/2))
		draw.BiLinear.Scale(dst, dst.Rect, img, img.Bounds(), draw.Over, nil)
		return dst
	}

	return img
}

func saveToPng(img image.Image, path string) error {
	if img == nil {
		return nil
	}

	if frameImgFd, err := os.Create(path); err != nil {
		return err
	} else {
		defer frameImgFd.Close()
		if err := png.Encode(frameImgFd, img); err != nil {
			return err
		}

		return nil
	}
}
