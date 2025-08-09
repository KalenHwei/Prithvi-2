export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --resume-download ibm-nasa-geospatial/Prithvi-EO-2.0-300M --local-dir /data6/personal/weiyongda/PhD/Weather/models/Prithvi-EO-2.0-300M &
huggingface-cli download --resume-download ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL --local-dir /data6/personal/weiyongda/PhD/Weather/models/Prithvi-EO-2.0-300M-TL &
huggingface-cli download --resume-download ibm-nasa-geospatial/Prithvi-EO-2.0-600M --local-dir /data6/personal/weiyongda/PhD/Weather/models/Prithvi-EO-2.0-600M &
huggingface-cli download --resume-download ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL --local-dir /data6/personal/weiyongda/PhD/Weather/models/Prithvi-EO-2.0-600M-TL &

wait
echo "所有模型下载完成！"