{$R-}{$I+}
var c:char;x:longint;
  d:array[0..1000000]of char;
begin
//Drive check and count
  for c:='C' to 'I' do
  begin
    mkdir(c+':\ccc');
    if ioresult<>0 then rmdir(c+':\ccc');
    if ioresult=0 then
    begin
      inc(x);d[x]:=c;writeln(c)
    end;
  end;
//Processing
{$I-}
  while true do
  for x:=1 to x do;
  readln
end.
