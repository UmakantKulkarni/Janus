#!/usr/bin/env bash

declare -a experiments=(
    "multi-pdu-session"
    "ngap-handover"
    "xn-handover"
    "multi-ue-loop"
    "amf-load-loop"
    "amf-availability"
)

NAMESPACE="open5gs"
PACKETRUSHER_BIN="/opt/PacketRusher/packetrusher"
PACKETRUSHER_CONFIG="/opt/PacketRusher/config/config.yml"
RESULT_ROOT="/opt/packetrusher_experiments"

mkdir -p "${RESULT_ROOT}"

for exprname in "${experiments[@]}"
do
    echo "Running experiment: ${exprname}"
    mkdir -p "${RESULT_ROOT}/${exprname}"
    rm -rf "${RESULT_ROOT}/${exprname}"/*.log

    kubectl get pods -n $NAMESPACE --no-headers=true | awk '/upf|amf|bsf|pcf|udm|ausf|nrf|nssf|udr|smf|scp/{print $1}'| xargs  kubectl delete pod -n $NAMESPACE
    sleep 10
    
    while [[ $(kubectl get pods -n $NAMESPACE -o 'jsonpath={..status.conditions[?(@.type=="Ready")].status}' | tr ' ' '\n' | grep -c "False") -ne 0 ]]; do
        echo "Waiting for pods to become ready..."
        sleep 5
    done

    status=1
    if [[ "${exprname}" == "multi-pdu-session" ]]; then
        "${PACKETRUSHER_BIN}" --config "${PACKETRUSHER_CONFIG}" multi-ue --number-of-ues 2 --numPduSessions 3 --timeBeforeIdle 10000 > "${RESULT_ROOT}/${exprname}/packetrusher.log" 2>&1 &
        sleep 5
        ps -ef | grep "${PACKETRUSHER_BIN}" | grep -v grep
        status=$?
    elif [[ "${exprname}" == "ngap-handover" ]]; then
        "${PACKETRUSHER_BIN}" --config "${PACKETRUSHER_CONFIG}" multi-ue -n 1 --timeBeforeNgapHandover 5000 > "${RESULT_ROOT}/${exprname}/packetrusher.log" 2>&1 &
        sleep 5
        ps -ef | grep "${PACKETRUSHER_BIN}" | grep -v grep
        status=$?
    elif [[ "${exprname}" == "xn-handover" ]]; then
        "${PACKETRUSHER_BIN}" --config "${PACKETRUSHER_CONFIG}" multi-ue -n 1 --timeBeforeXnHandover 5000 > "${RESULT_ROOT}/${exprname}/packetrusher.log" 2>&1 &
        sleep 5
        ps -ef | grep "${PACKETRUSHER_BIN}" | grep -v grep
        status=$?
    elif [[ "${exprname}" == "multi-ue-loop" ]]; then
        "${PACKETRUSHER_BIN}" --config "${PACKETRUSHER_CONFIG}" multi-ue --number-of-ues 5 --loop --loopCount 3 --timeBeforeDeregistration 3000 --timeBeforeReregistration 1000 > "${RESULT_ROOT}/${exprname}/packetrusher.log" 2>&1 &
        sleep 5
        ps -ef | grep "${PACKETRUSHER_BIN}" | grep -v grep
        status=$?
    elif [[ "${exprname}" == "amf-load-loop" ]]; then
        "${PACKETRUSHER_BIN}" --config "${PACKETRUSHER_CONFIG}" amf-load-loop -n 20 -t 30 > "${RESULT_ROOT}/${exprname}/packetrusher.log" 2>&1 &
        sleep 5
        ps -ef | grep "${PACKETRUSHER_BIN}" | grep -v grep
        status=$?
    elif [[ "${exprname}" == "amf-availability" ]]; then
        "${PACKETRUSHER_BIN}" --config "${PACKETRUSHER_CONFIG}" amf-availability -t 25 > "${RESULT_ROOT}/${exprname}/packetrusher.log" 2>&1 &
        sleep 5
        ps -ef | grep "${PACKETRUSHER_BIN}" | grep -v grep
        status=$?
    else
        echo "Unknown experiment ${exprname}."
    fi

    echo "Waiting for 60 seconds to run experiment..."
    sleep 60
    pkill -f ${PACKETRUSHER_BIN}
    sleep 10

    for pod in $(kubectl -n "$NAMESPACE" get po -o json | jq '.items[] | select(.metadata.name|contains("open5gs"))| .metadata.name' | grep -v "test\|webui\|mongo" | sed 's/"//g') ;
    do
        echo "$pod"
        kubectl -n "$NAMESPACE" logs "$pod" > "${RESULT_ROOT}/${exprname}/${pod}.log"
    done

    if [[ $status -eq 0 ]]; then
        echo "Experiment ${exprname} PacketRusher succeeded."
    else
        echo "Experiment ${exprname} PacketRusher failed."
    fi

    if grep -q "ERROR" "${RESULT_ROOT}/${exprname}"/*-amf-*.log; then
        echo "Experiment ${exprname} failed."
    else
        echo "Experiment ${exprname} succeeded."
    fi

    echo "Experiment ${exprname} completed."
    pkill -f ${PACKETRUSHER_BIN}
done

echo "All experiments completed."