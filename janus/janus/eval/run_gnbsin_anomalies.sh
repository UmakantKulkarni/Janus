#!/usr/bin/env bash

declare -a experiments=("security-complete-before-command" "malformed-nas" "empty-nas"  "deregistration-before-registration" "auth-response-before-request")

NAMESPACE="open5gs"

mkdir -p /opt/gnbsim_experiments

for exprname in "${experiments[@]}"
do
    echo "Running experiment: ${exprname}"
    mkdir -p /opt/gnbsim_experiments/${exprname}
    rm -rf /opt/gnbsim_experiments/${exprname}/*.log

    kubectl get pods -n $NAMESPACE --no-headers=true | awk '/upf|amf|bsf|pcf|udm|ausf|nrf|nssf|udr|smf|scp/{print $1}'| xargs  kubectl delete pod -n $NAMESPACE
    sleep 10
    
    while [[ $(kubectl get pods -n $NAMESPACE -o 'jsonpath={..status.conditions[?(@.type=="Ready")].status}' | tr ' ' '\n' | grep -c "False") -ne 0 ]]; do
        echo "Waiting for pods to become ready..."
        sleep 5
    done

    /opt/gnbsim/gnbsim --cfg /opt/gnbsim/anomaly/${exprname}.yaml > /opt/gnbsim_experiments/${exprname}/gnb.log

    sleep 10

    for pod in `kubectl -n $NAMESPACE get po -o json |  jq '.items[] | select(.metadata.name|contains("open5gs"))| .metadata.name' | grep -v "test\|webui\|mongo" | sed 's/"//g'` ;
    do
        echo $pod
        kubectl -n $NAMESPACE logs $pod > /opt/gnbsim_experiments/${exprname}/${pod}.log
    done

    if grep -q "Ue's Passed: 1" /opt/gnbsim_experiments/${exprname}/gnb.log; then
        echo "Experiment ${exprname} gNBSim succeeded."
    else
        echo "Experiment ${exprname} gNBSim failed."
    fi

    if grep -q "ERROR" /opt/gnbsim_experiments/${exprname}/*-amf-*.log; then
        echo "Experiment ${exprname} AMF failure succeeded."
    else
        echo "Experiment ${exprname} AMF failure failed."
    fi

    echo "Experiment ${exprname} completed."
done

echo "All experiments completed."
