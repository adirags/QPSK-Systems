%% Initializations

FRM = 128;
MaxNumErrs = 200;
MaxNumBits = 1e7;
EbNo_vector = 0:10;
BER_vectorHard = zeros(size(EbNo_vector));
BER_vectorSoft = zeros(size(EbNo_vector));
Modulator = comm.QPSKModulator('BitInput',true);
AWGN = comm.AWGNChannel;
HardDemodulator = comm.QPSKDemodulator('BitOutput',true);
SoftDemodulator = comm.QPSKDemodulator('BitOutput',true,'DecisionMethod','Log-likelihood ratio','VarianceSource','Input port');
BitError = comm.ErrorRate;
ConvEncoder = comm.ConvolutionalEncoder('TerminationMethod','Terminated');
ViterbiHard = comm.ViterbiDecoder('InputFormat','Hard','TerminationMethod','Terminated');
ViterbiSoft = comm.ViterbiDecoder('InputFormat','Soft','SoftInputWordLength', 4,'OutputDataType', 'double','TerminationMethod','Terminated');
Quantizer = dsp.ScalarQuantizerEncoder('Partitioning','Unbounded','BoundaryPoints',-7:7,'OutputIndexDataType','uint8');
M = 4;
k = log2(M);
CodeRate = 1/2;
%% Processsing loop modeling transmitter, channel model and receiver


for EbNo = EbNo_vector
    snr = EbNo + 10*log10(k) + 10*log10(CodeRate);
    noise_var = 10.^(snr/10);
    AWGN.EbNo = snr;
    numErrsHard = 0; 
    numBitsHard = 0;
    numErrsSoft = 0; 
    numBitsSoft = 0;
    resultsHard = zeros(3,1);
    resultsSoft = zeros(3,1);

    %% Using Viterbi Hard Decision

    while ((numErrsHard < MaxNumErrs) && (numBitsHard < MaxNumBits))
        %Transmitter
        u = randi([0 1], FRM,1); % Random bits generator
        encoded = ConvEncoder.step(u); % Convolutional encoder
        mod_sig = Modulator.step(encoded); % QPSK Modulator
        % Channel
        rx_sig = AWGN.step(mod_sig); % AWGN channel
        % Receiver
        demod = HardDemodulator.step(rx_sig); % QPSK Demodulator
        HardDecoded = ViterbiHard.step(demod); % Hard Viterbi decoder
        HardY = HardDecoded(1:FRM); % Compute output bits from hard decode
        resultsHard = BitError.step(u, HardY); % Update BER
        numErrsHard = resultsHard(2);
        numBitsHard = resultsHard(3);
    end
    
    %% Clean up & collect results
    
    berHard = resultsHard(1); 
    bitsHard = resultsHard(3);
    reset(BitError);
    BER_vectorHard(EbNo+1)=berHard;
   
    %% Using Soft Decision
    
    while ((numErrsSoft < MaxNumErrs) && (numBitsSoft < MaxNumBits))
        %Transmitter
        u = randi([0 1], FRM,1); % Random bits generator
        encoded = ConvEncoder.step(u); % Convolutional encoder
        mod_sig = Modulator.step(encoded); % QPSK Modulator
        % Channel
        rx_sig = AWGN.step(mod_sig); % AWGN channel
        % Receiver
        demod = SoftDemodulator.step(rx_sig,noise_var); % Soft Decision QPSK Demodulator
        llr = Quantizer.step(-demod); % Log-Likelihood Ratio Computation
        SoftDecoded = ViterbiSoft.step(llr); % Soft Viterbi decoder
        SoftY = SoftDecoded(1:FRM); % Compute output bits from soft decode
        resultsSoft = BitError.step(u, SoftY); % Update BER
        numErrsSoft = resultsSoft(2);
        numBitsSoft = resultsSoft(3);
    end
    
    %% Clean up & collect results
    
    berSoft = resultsSoft(1); 
    bitsSoft = resultsSoft(3);
    reset(BitError);
    BER_vectorSoft(EbNo+1)=berSoft;
end

%% Visualize results

EbNoLin = 10.^(EbNo_vector/10);
theoretical_results = 0.5*erfc(sqrt(EbNoLin));
semilogy(EbNo_vector, BER_vectorHard)
grid;title('BER vs. EbNo - QPSK modulation with Viterbi Hard Decoding');
xlabel('Eb/No (dB)');ylabel('BER');hold;
semilogy(EbNo_vector,theoretical_results,'dr');hold;
legend('Simulation - Viterbi Hard Decoding','Theoretical');

figure;
semilogy(EbNo_vector, BER_vectorSoft)
grid;title('BER vs. EbNo - QPSK modulation with Viterbi Soft Decoding');
xlabel('Eb/No (dB)');ylabel('BER');hold;
semilogy(EbNo_vector,theoretical_results,'dr');hold;
legend('Simulation - Viterbi Soft Decoding','Theoretical');
