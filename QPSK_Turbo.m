%% Initialization

FRM = 128;
maxNumErrs = 200;
maxNumBits = 1e7;
EbNo_vector = 1:10;
BER_vector = zeros(size(EbNo_vector));
Trellis = poly2trellis(4, [13 15], 13);
Indices = randperm(FRM);
M = 4;
k = log2(M);
R = FRM/(3* FRM + 4*3);
Modulator = comm.QPSKModulator('BitInput',true);
AWGN = comm.AWGNChannel;
Demodulator = comm.QPSKDemodulator('BitOutput',true,'DecisionMethod','Log-likelihood ratio','VarianceSource', 'Input port');
BitError = comm.ErrorRate;
TurboEncoder=comm.TurboEncoder('TrellisStructure',Trellis,'InterleaverIndices',Indices);
TurboDecoder=comm.TurboDecoder('TrellisStructure',Trellis,'InterleaverIndices',Indices,'NumIterations',6);

%% Processing Loop

for EbNo = EbNo_vector
    snr = EbNo + 10*log10(k) + 10*log10(R);
    noise_var = 10.^(-snr/10);
    AWGN.EbNo = snr;
    numErrs = 0; 
    numBits = 0; 
    results = zeros(3,1);
    while ((numErrs < maxNumErrs) && (numBits < maxNumBits))
        % Transmitter
        u = randi([0 1], FRM,1); % Random bits generator
        encoded = TurboEncoder.step(u); % Turbo Encoder
        mod_sig = Modulator.step(encoded); % QPSK Modulator
        % Channel
        rx_sig = AWGN.step(mod_sig); % AWGN channel
        % Receiver
        demod = Demodulator.step(rx_sig, noise_var); %Soft-decisionQPSK Demodulator
        decoded = TurboDecoder.step(-demod); % Turbo Decoder
        y = decoded(1:FRM); % Compute output bits
        results = BitError.step(u, y); % Update BER
        numErrs = results(2);
        numBits = results(3);
    end

    %% Clean up & collect results

    ber = results(1); 
    bits= results(3);
    reset(BitError);
    BER_vector(EbNo+1) = ber;
end
%% Visualize results

EbNoLin = 10.^(EbNo_vector/10);
theoretical_results = 0.5*erfc(sqrt(EbNoLin));
semilogy(EbNo_vector, BER_vector);
grid;title('BER vs. EbNo - QPSK modulation with Turbo Encoding');
xlabel('Eb/No (dB)');ylabel('BER');hold;
semilogy(EbNo_vector,theoretical_results,'dr');hold;
legend('Simulation - Turbo Encoded','Theoretical');