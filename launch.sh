#!/bin/bash
echo "Mariana launch time: $(date)"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# add lucifer to PYTHONPATH, also add cruise to PYTHONPATH if using scm
# export PYTHONPATH=$SCRIPT_DIR:/opt/tiger/cruise:/opt/tiger/Megatron-LM/:/opt/tiger/mariana_metadata:/opt/tiger/janus/:/opt/tiger/flux/lib/:/opt/tiger/flux/python/:/opt/tiger/TransformerEngine:$PYTHONPATH
# export LD_LIBRARY_PATH=/opt/tiger/flux/lib:$LD_LIBRARY_PATH

# prefetch ckpt
# python3 mariana/utils/prefetch_ckpt.py $@ &

# pre-compilation of JIT kernels
# python3 precompile_job.py &

# bash setup_cruise.sh
# if [ "$MARIANA_USE_BPEX_KERNEL" == "1" ]
# then
#     bash setup_bpex.sh
# fi

# set default communication env vars
export BYTED_TORCH_BYTECCL=O0
export MARIANA_CPU_EXCLUSIVE_CORES=${MARIANA_CPU_EXCLUSIVE_CORES:=2}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:=23}
export MARIANA_USE_NUMACTL=${MARIANA_USE_NUMACTL:=1}

if [ "$MARIANA_USE_NUMACTL" == "1" ] && [ "$(which numactl)" == "" ];
then
    echo "Warning: unable to find numactl command"
    export MARIANA_USE_NUMACTL=0
fi

# disable tcc fetching and usage report in cruise by default
export CRUISE_DISABLE_USAGE_REPORT=${CRUISE_DISABLE_USAGE_REPORT:=True}
export CRUISE_DISABLE_TCC_FETCH=${CRUISE_DISABLE_TCC_FETCH:=True}

# enable partial encryption, which only turns on encryption on worker_0
if [ "$MARIANA_ENABLE_PARTIAL_ENCRYPTION" == "1" ]
then
    if [ $ARNOLD_ID == "0"]; then
        echo "PARTIAL ENCRYPTION is ON, this is worker 0 and enable encryption now"
        export KMS_ZONE=cn
        export DKMS_ZONE=lf  #使用 LF 的机房服务
        export NO_PROXY=localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8
        export TRAIN_ENABLE_UPLOAD_ENCRYPT=1
    else
        echo "PARTIAL ENCRYPTION is ON, this is worker_$ARNOLD_ID, skip encryption on this worker"
    fi
else
    echo "PARTIAL ENCRYPTION is OFF"
fi

if [ "$MARIANA_ENABLE_FULL_ENCRYPTION" == "1" ]
then
    echo "FULL ENCRYPTION is ON, turn on uploading encryption anyway"
    export KMS_ZONE=cn
    export DKMS_ZONE=lf  #使用 LF 的机房服务
    export NO_PROXY=localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8
    export TRAIN_ENABLE_UPLOAD_ENCRYPT=1
else
    echo "FULL ENCRYPTION is OFF, do nothing here"
fi

# show all configs with defaults and user override first
# echo "==<Displaying CLI options>=="
# python3 $@ --print_config
# echo "==<End of displaying CLI options>=="

# run it
echo "==<Using cruise included TORCHRUN>=="
#--rdzv_endpoint=127.0.0.1:30000
# ./TORCHRUN  $@

if [[ "$ENABLE_NSYS" == "1" ]]
then
    (
      cd /opt/tiger && wget http://tosv.byted.org/obj/bin/toscli -O toscli && chmod a+x toscli && cd -
      export https_proxy=http://bj-rd-proxy.byted.org:3128
      curl -fsSL https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | sudo apt-key add - && \
          echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > cuda.list
      sudo mv cuda.list /etc/apt/sources.list.d/
      sudo -E apt-get update -y && sudo -E apt-get -y install cuda-nsight-systems-12-2 --no-install-recommends
    )
    nsys start -c cudaProfilerApi -o worker${ARNOLD_ID}_nsys_trace.nsys-rep
    nsys launch --wait primary TORCHRUN $@
else
    ./TORCHRUN $@
fi

ret=$?

echo "Mariana exit time: $(date)"

exit $ret
