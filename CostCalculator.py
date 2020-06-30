
def BridgeCost(SlopD, Weight, AgrD, NP, Cost):
    for x in range(NP):
        if Weight[x] <= 175:
            Cost[x] = SlopD * 3150000 * AgrD[x]
        elif Weight[x] > 300:
            Cost[x] = 16000 * (Weight[x] - 237.5) + SlopD * 3150000 * AgrD[x]
        else:
            Cost[x] = 8000 * (Weight[x] - 175) + SlopD * 3150000 * AgrD[x]
    return Cost
