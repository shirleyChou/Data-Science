function count = computeOne(idx)
count=0;
for i=1:length(idx),
if idx(i) == 1,
count = count + 1;
end;
end;